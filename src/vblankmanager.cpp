// Try to figure out when vblank is and notify steamcompmgr to render some time before it

#include <mutex>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <condition_variable>

#include <assert.h>
#include <fcntl.h>
#include <unistd.h>
#include <vulkan/vulkan_core.h>

#include "gpuvis_trace_utils.h"

#include "vblankmanager.hpp"
#include "steamcompmgr.hpp"
#include "wlserver.hpp"
#include "main.hpp"

static int g_vblankPipe[2];

std::atomic<uint64_t> g_lastVblank;

// 3ms by default -- a good starting value.
const uint64_t g_uStartingDrawTime = 3'000'000;

// This is the last time a draw took.
std::atomic<uint64_t> g_uVblankDrawTimeNS = { g_uStartingDrawTime };

// Tuneable
// 2.0ms by default. (g_DefaultVBlankRedZone)
// This is the leeway we always apply to our buffer.
// This also accounts for some time we cannot account for (which (I think) is the drm_commit -> triggering the pageflip)
// It would be nice to make this lower if we can find a way to track that effectively
// Perhaps the missing time is spent elsewhere, but given we track from the pipe write
// to after the return from `drm_commit` -- I am very doubtful.
uint64_t g_uVblankDrawBufferRedZoneNS = g_uDefaultVBlankRedZone;

// Tuneable
// 93% by default. (g_uVBlankRateOfDecayPercentage)
// The rate of decay (as a percentage) of the rolling average -> current draw time
uint64_t g_uVBlankRateOfDecayPercentage = g_uDefaultVBlankRateOfDecayPercentage;

const uint64_t g_uVBlankRateOfDecayMax = 100;

//#define VBLANK_DEBUG

void vblankThreadRun( void )
{
	pthread_setname_np( pthread_self(), "gamescope-vblk" );

	// Start off our average with our starting draw time.
	uint64_t rollingMaxDrawTime = g_uStartingDrawTime;

	const uint64_t range = g_uVBlankRateOfDecayMax;
	while ( true )
	{
		const uint64_t alpha = g_uVBlankRateOfDecayPercentage;
		const int refresh = g_nNestedRefresh ? g_nNestedRefresh : g_nOutputRefresh;

		const uint64_t nsecInterval = 1'000'000'000ul / refresh;
		const uint64_t drawTime = g_uVblankDrawTimeNS;

		// This is a rolling average when drawTime < rollingMaxDrawTime,
		// and a a max when drawTime > rollingMaxDrawTime.
		// This allows us to deal with spikes in the draw buffer time very easily.
		// eg. if we suddenly spike up (eg. because of test commits taking a stupid long time),
		// we will then be able to deal with spikes in the long term, even if several commits after
		// we get back into a good state and then regress again.
		rollingMaxDrawTime = ( ( alpha * std::max( rollingMaxDrawTime, drawTime ) ) + ( range - alpha ) * drawTime ) / range;

		// If we need to offset for our draw more than half of our vblank, something is very wrong.
		// Clamp our max time to half of the vblank if we can.
		rollingMaxDrawTime = std::min( rollingMaxDrawTime + g_uVblankDrawBufferRedZoneNS, nsecInterval / 2 ) - g_uVblankDrawBufferRedZoneNS;

		uint64_t offset = rollingMaxDrawTime + g_uVblankDrawBufferRedZoneNS;

#ifdef VBLANK_DEBUG
		// Debug stuff for logging missed vblanks
		static uint64_t vblankIdx = 0;
		static uint64_t lastDrawTime = g_uVblankDrawTimeNS;
		static uint64_t lastOffset = g_uVblankDrawTimeNS + g_uVblankDrawBufferRedZoneNS;

		if ( vblankIdx++ % 300 == 0 || drawTime > lastOffset )
		{
			if ( drawTime > lastOffset )
				fprintf( stderr, " !! missed vblank " );

			fprintf( stderr, "redZone: %.2fms decayRate: %lu%% - rollingMaxDrawTime: %.2fms lastDrawTime: %.2fms lastOffset: %.2fms - drawTime: %.2fms offset: %.2fms\n",
				g_uVblankDrawBufferRedZoneNS / 1'000'000.0,
				g_uVBlankRateOfDecayPercentage,
				rollingMaxDrawTime / 1'000'000.0,
				lastDrawTime / 1'000'000.0,
				lastOffset / 1'000'000.0,
				drawTime / 1'000'000.0,
				offset / 1'000'000.0 );
		}

		lastDrawTime = drawTime;
		lastOffset = offset;
#endif

		uint64_t lastVblank = g_lastVblank - offset;

		uint64_t now = get_time_in_nanos();
		uint64_t targetPoint = lastVblank + nsecInterval;
		while ( targetPoint < now )
			targetPoint += nsecInterval;

		sleep_until_nanos( targetPoint );

		// give the time of vblank to steamcompmgr
		uint64_t vblanktime = get_time_in_nanos();

		ssize_t ret = write( g_vblankPipe[ 1 ], &vblanktime, sizeof( vblanktime ) );
		if ( ret <= 0 )
		{
			perror( "vblankmanager: write failed" );
		}
		else
		{
			gpuvis_trace_printf( "sent vblank" );
		}
		
		// Get on the other side of it now
		sleep_for_nanos( offset + 1'000'000 );
	}
}

int vblank_init( void )
{
	if ( pipe2( g_vblankPipe, O_CLOEXEC | O_NONBLOCK ) != 0 )
	{
		perror( "vblankmanager: pipe failed" );
		return -1;
	}
	
	g_lastVblank = get_time_in_nanos();

	std::thread vblankThread( vblankThreadRun );
	vblankThread.detach();

	return g_vblankPipe[ 0 ];
}

void vblank_mark_possible_vblank( uint64_t nanos )
{
	g_lastVblank = nanos;
}

// fps limit manager

static std::mutex g_TargetFPSMutex;
static std::condition_variable g_TargetFPSCondition;
static int g_nFpsLimitTargetFPS = 0;
struct FrameInfo_t
{
	uint64_t lastFrame;
	uint64_t currentFrame;
	uint64_t frameCount;
};

FrameInfo_t g_FrameInfo;

void steamcompmgr_fpslimit_release_commit();
void steamcompmgr_send_frame_done_to_focus_window();

//#define FPS_LIMIT_DEBUG

void fpslimitThreadRun( void )
{
	pthread_setname_np( pthread_self(), "gamescope-fps" );

	uint64_t deviation = 0;
	uint64_t lastFrameCount = 0;
	uint64_t lastCommitReleased = get_time_in_nanos();
	while ( true )
	{
		FrameInfo_t frameInfo;
		int nTargetFPS;
		{
			std::unique_lock<std::mutex> lock( g_TargetFPSMutex );
			g_TargetFPSCondition.wait(lock, [lastFrameCount]{ return g_nFpsLimitTargetFPS != 0 && g_FrameInfo.frameCount != lastFrameCount; });
			nTargetFPS = g_nFpsLimitTargetFPS;
			frameInfo = g_FrameInfo;
		}

		// Check if we are unaligned or not, as to whether
		// we call frame callbacks from this thread instead of steamcompmgr based
		// on vblank count.
		bool useFrameCallbacks = fpslimit_use_frame_callbacks_for_focus_window( nTargetFPS, 0 );

		uint64_t targetInterval = 1'000'000'000ul / nTargetFPS;

		uint64_t t0 = lastCommitReleased;
		uint64_t t1 = get_time_in_nanos();
	
		// Not the actual frame time of the game
		// this is the time of the amount of work a 'frame' has done.
		uint64_t frameTime = t1 - t0;
		lastFrameCount = frameInfo.frameCount;

#ifdef FPS_LIMIT_DEBUG
		fprintf( stderr, "frame time = %.2fms - target %.2fms - deviation %.2fms \n", frameTime / 1'000'000.0, targetInterval / 1'000'000.0, deviation / 1'000'000.0 );
#endif

		if ( frameTime * 100 > targetInterval * 103 - deviation * 100 )
		{
			// If we have a slow frame, reset the deviation since we
			// do not want to compensate for low performance later on
			deviation = 0;
			steamcompmgr_fpslimit_release_commit();
			lastCommitReleased = get_time_in_nanos();

			// If we aren't vblank aligned, send our frame callbacks here.
			if ( !useFrameCallbacks )
				steamcompmgr_send_frame_done_to_focus_window();
		}
		else
		{
			uint64_t now = get_time_in_nanos();

			uint64_t targetPoint = now + targetInterval - deviation - frameTime;

			while ( targetPoint < now )
				targetPoint += targetInterval;

			//fprintf( stderr, "Sleeping from %lu to %lu to reach %d fps\n", now, targetPoint, g_nFpsLimitTargetFPS );
			sleep_until_nanos( targetPoint );
			t1 = get_time_in_nanos();

			frameTime = t1 - t0;
			deviation += frameTime - targetInterval;
			deviation = std::min( deviation, targetInterval / 16 );

			steamcompmgr_fpslimit_release_commit();
			lastCommitReleased = get_time_in_nanos();

			// If we aren't vblank aligned, send our frame callbacks here.
			if ( !useFrameCallbacks )
				steamcompmgr_send_frame_done_to_focus_window();
		}

		// If we aren't vblank aligned, nudge ourselves to process done commits now.
		if ( !useFrameCallbacks )
		{
			steamcompmgr_send_frame_done_to_focus_window();
			nudge_steamcompmgr();
		}
	}
}

void fpslimit_init( void )
{
	g_FrameInfo.lastFrame = get_time_in_nanos();
	g_FrameInfo.currentFrame = get_time_in_nanos();

	std::thread fpslimitThread( fpslimitThreadRun );
	fpslimitThread.detach();
}

void fpslimit_mark_frame( void )
{
	std::unique_lock<std::mutex> lock( g_TargetFPSMutex );
	{
		g_FrameInfo.lastFrame = g_FrameInfo.currentFrame;
		g_FrameInfo.currentFrame = get_time_in_nanos();
		g_FrameInfo.frameCount++;
	}
	g_TargetFPSCondition.notify_all();
}

bool fpslimit_use_frame_callbacks_for_focus_window( int nTargetFPS, int nVBlankCount ) 
{
	if ( !nTargetFPS )
		return true;

	if ( g_nOutputRefresh % nTargetFPS == 0 )
	{
		// Aligned, limit based on vblank count.
		return nVBlankCount % ( g_nOutputRefresh / nTargetFPS );
	}
	else
	{
		// Unaligned from VBlank, never use frame callbacks on SteamCompMgr thread.
		// call them from fpslimit
		return false;
	}
}

// Called from steamcompmgr thread
void fpslimit_set_target( int nTargetFPS )
{
	{
		std::unique_lock<std::mutex> lock(g_TargetFPSMutex);
		g_nFpsLimitTargetFPS = nTargetFPS;
	}

	g_TargetFPSCondition.notify_all();
}
