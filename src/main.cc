#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/poll.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <dirent.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <insightface/inspireface.h>

#include "rtsp_demo.h"
#include "luckfox_mpi.h"

#define DISP_WIDTH                  1280    // 720
#define DISP_HEIGHT                 1080    // 480

MPP_CHN_S stSrcChn, stSrcChn1, stvpssChn, stvencChn;
VENC_RECV_PIC_PARAM_S stRecvParam;

rtsp_demo_handle g_rtsplive = NULL;
rtsp_session_handle g_rtsp_session;

HFSession g_face_session;

void LoadReferenceFace()
{
    printf("LoadReferenceFace\n");
    const char* face_path = "./faces";
    DIR *dir = opendir(face_path);
    if (dir == NULL) {
        return;
    }

    struct dirent *entry;
    int face_idx = 0;
    
    char path[256] = {0};
    char cstr[100] = {0};

    int ret;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(".", entry->d_name) == 0 || strcmp("..", entry->d_name) == 0) {
            continue;
        }

        sprintf(path, "%s/%s", face_path, entry->d_name);
        printf("face feature name: %s\n", path);

        cv::Mat image = cv::imread(path);
        if (image.empty()) {
            printf("Face Image is empty: %s\n", entry->d_name);
            continue;
        }

        HFImageData imageData = {0};
        imageData.data = image.data; // Pointer to the image data
        imageData.format = HF_STREAM_BGR; // Image format (BGR in this case)
        imageData.height = image.rows; // Image height
        imageData.width = image.cols; // Image width
        imageData.rotation = HF_CAMERA_ROTATION_0; // Image rotation
        HFImageStream stream;
        ret = HFCreateImageStream(&imageData, &stream); // Create an image stream for processing
        if (ret != HSUCCEED) {
            printf("Create stream error: %d\n", ret);
            continue;
        }

        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(g_face_session, stream, &multipleFaceData); // Track faces in the image
        if (ret != HSUCCEED) {
            printf("Run face track error: %d\n", ret);
            continue;
        }
        if (multipleFaceData.detectedNum == 0) { // Check if any faces were detected
            printf("No face was detected: %s, %d\n", entry->d_name, ret);
            continue;
        }

        HFFaceFeature feature = {0};
        ret = HFFaceFeatureExtract(g_face_session, stream, multipleFaceData.tokens[0], &feature); // Extract features
        if (ret != HSUCCEED) {
            printf("Extract feature error: %d\n", ret);
            continue;
        }

        memcpy(cstr, entry->d_name, strlen(entry->d_name) - 4);
        HFFaceFeatureIdentity identity = {0};
        identity.feature = &feature; // Assign the extracted feature
        identity.customId = face_idx; // Custom identifier for the face
        identity.tag = cstr; // Tag the feature with the name
        ret = HFFeatureHubInsertFeature(identity); // Insert the feature into the hub
        if (ret != HSUCCEED) {
            printf("Feature insertion into FeatureHub failed: %d\n", ret);
            continue;
        }
        
        ret = HFReleaseImageStream(stream);
        if (ret != HSUCCEED) {
            printf("Release image stream error: %lu\n", ret);
        }

        face_idx++;
    }

    closedir(dir);

    HInt32 count;
    ret = HFFeatureHubGetFaceCount(&count);
    printf("Inserted data: %d\n", count);
    printf("LoadReferenceFace finished, %d faces loaded.\n", count);
}

static void* GetMediaBuffer(void *arg) {
    (void)arg;
    printf("========%s========\n", __func__);
    void *pData = RK_NULL;

    int s32Ret;

    VENC_STREAM_S stFrame;
    stFrame.pstPack = (VENC_PACK_S *) malloc(sizeof(VENC_PACK_S));

    while (1) {
        s32Ret = RK_MPI_VENC_GetStream(0, &stFrame, -1);
        if (s32Ret == RK_SUCCESS) {
            if (g_rtsplive && g_rtsp_session) {
                pData = RK_MPI_MB_Handle2VirAddr(stFrame.pstPack->pMbBlk);
                rtsp_tx_video(g_rtsp_session, (uint8_t *)pData, stFrame.pstPack->u32Len, stFrame.pstPack->u64PTS);

                rtsp_do_event(g_rtsplive);
            }

            s32Ret = RK_MPI_VENC_ReleaseStream(0, &stFrame);
            if (s32Ret != RK_SUCCESS) {
                RK_LOGE("RK_MPI_VENC_ReleaseStream fail %x", s32Ret);
            }
        }
        usleep(10 * 1000);
    }
    printf("\n======exit %s=======\n", __func__);
    free(stFrame.pstPack);

    return NULL;
}

static void *RetinaProcessBuffer(void *arg) {
    (void)arg;
    printf("========%s========\n", __func__);

    int disp_width  = DISP_WIDTH;
	int disp_height = DISP_HEIGHT;
	
	char text[16];	
	int sX,sY,eX,eY;
	int s32Ret;
	int group_count = 0;
	VIDEO_FRAME_INFO_S stViFrame;
	
    while(1)
	{
		s32Ret = RK_MPI_VI_GetChnFrame(0, 1, &stViFrame, -1);
		if(s32Ret == RK_SUCCESS)
		{
			void *vi_data = RK_MPI_MB_Handle2VirAddr(stViFrame.stVFrame.pMbBlk);
			if(vi_data != RK_NULL)
			{
				cv::Mat yuv420sp(disp_height + disp_height / 2, disp_width, CV_8UC1, vi_data);
				cv::Mat bgr(disp_height, disp_width, CV_8UC3);			
				cv::cvtColor(yuv420sp, bgr, cv::COLOR_YUV420sp2BGR);

                HFImageData imageData = {0};
                imageData.data = bgr.data;
                imageData.format = HF_STREAM_BGR;
                imageData.height = bgr.rows;
                imageData.width = bgr.cols;
                imageData.rotation = HF_CAMERA_ROTATION_0;
                
                HFImageStream stream;
                HResult ret = HFCreateImageStream(&imageData, &stream);
                if (ret != HSUCCEED) {
                    printf("Create stream error: %d\n", ret);
                    continue;
                }

                HFMultipleFaceData multipleFaceData = {0};
                ret = HFExecuteFaceTrack(g_face_session, stream, &multipleFaceData);
                if (ret != HSUCCEED) {
                    printf("Run face track error: %d\n", ret);
                    HFReleaseImageStream(stream);
                    continue;
                }

                for (int i = 0; i < multipleFaceData.detectedNum; i++) {
                    printf("Face Index: %d, conf: %f\n", i, multipleFaceData.detConfidence[i]);

                    float face_conf = multipleFaceData.detConfidence[i];
                    if (face_conf < 0.5) {
                        continue;
                    }

                    sX = multipleFaceData.rects[i].x;	
                    sY = multipleFaceData.rects[i].y;	
                    eX = sX + multipleFaceData.rects[i].width;	
                    eY = sY + multipleFaceData.rects[i].height;

                    sX = sX - (sX % 2);
                    sY = sY - (sY % 2);
                    eX = eX	- (eX % 2);
                    eY = eY	- (eY % 2);

                    test_rgn_overlay_line_process(sX, sY, 0, group_count);
                    test_rgn_overlay_line_process(eX, sY, 1, group_count);
                    test_rgn_overlay_line_process(eX, eY, 2, group_count);
                    test_rgn_overlay_line_process(sX, eY, 3, group_count);

                    // Initialize the feature structure to store extracted face features
                    HFFaceFeature feature = {0};
                    
                    // // Extract facial features from the detected face using the first token
                    ret = HFFaceFeatureExtract(g_face_session, stream, multipleFaceData.tokens[i], &feature);
                    if (ret != HSUCCEED) {
                        printf("Extract feature error: %d\n", ret); // Print error if extraction fails
                        continue;
                    }

                    HFFaceFeatureIdentity searched = {0};
                    HFloat confidence; 
                    
                    ret = HFFeatureHubFaceSearch(feature, &confidence, &searched);
                    if (ret != HSUCCEED) {
                        printf("Search face feature error: %d\n", ret); // Print error if search fails
                        continue;
                    }
                    printf("Searched size: %d\n", searched.customId);

                    if (searched.customId != -1) {
                        sprintf(text, "%s(%.2f)", searched.tag, confidence);
                        test_rgn_overlay_text_process(sX + 4, sY + 4, text, group_count, 1);

                        printf("Found similar face: id= %d, tag=%s, confidence=%f\n", searched.customId, searched.tag, confidence);

                        // TODO: Open the door, insert record to database.
                    } else {
                        printf("Not Found similar\n");
                        sprintf(text, "%s", "Unregistered.");
                        test_rgn_overlay_text_process(sX + 4, sY + 4, text, group_count, 0);
                    }

                    group_count++;
                }

                ret = HFReleaseImageStream(stream);
                if (ret != HSUCCEED) {
                    printf("Release stream error: %d\n", ret);
                }
			}

			s32Ret = RK_MPI_VI_ReleaseChnFrame(0, 1, &stViFrame);
			if (s32Ret != RK_SUCCESS) {
				RK_LOGE("RK_MPI_VI_ReleaseChnFrame fail %x", s32Ret);
			}
		}
		else{
			printf("Get viframe error %d !\n", s32Ret);
			continue;
		}

		usleep(500000);
        for(int i = 0;i < group_count; i++) {
            rgn_overlay_release(i);
        }
        group_count = 0;
	}			
	
    return NULL;
}

int main(int argc, char* argv[])
{
    system("RkLunch-stop.sh");
    RK_S32 s32Ret = 0;

    int width       = DISP_WIDTH;
    int height      = DISP_HEIGHT;

    const char *model_path = "./model/Pikachu";
    HResult ret = HFLaunchInspireFace(model_path);
    if (ret != HSUCCEED) {
        printf("Load Model error: %d\n", ret);
        return ret;
    }

    HFFeatureHubConfiguration featureHubConfiguration;
    featureHubConfiguration.featureBlockNum = 10; // Number of feature blocks
    featureHubConfiguration.enablePersistence = 0; // Persistence not enabled, use in-memory database
    featureHubConfiguration.dbPath = ""; // Database path (not used here)
    featureHubConfiguration.searchMode = HF_SEARCH_MODE_EAGER; // Search mode configuration
    featureHubConfiguration.searchThreshold = 0.48f; // Threshold for search operations

    // Enable the global feature database
    ret = HFFeatureHubDataEnable(featureHubConfiguration);
    if (ret != HSUCCEED) {
        printf("An exception occurred while starting FeatureHub: %d\n", ret);
        return ret;
    }

    HOption option = HF_ENABLE_FACE_RECOGNITION;
    ret = HFCreateInspireFaceSessionOptional(option, HF_DETECT_MODE_ALWAYS_DETECT, 1, -1, -1, &g_face_session);
    if (ret != HSUCCEED) {
        printf("Create session error: %d\n", ret);
        return ret;
    }

    LoadReferenceFace();

    // rkaiq init 
	RK_BOOL multi_sensor = RK_FALSE;	
	const char *iq_dir = "/etc/iqfiles";
	rk_aiq_working_mode_t hdr_mode = RK_AIQ_WORKING_MODE_NORMAL;
	//hdr_mode = RK_AIQ_WORKING_MODE_ISP_HDR2;
	SAMPLE_COMM_ISP_Init(0, hdr_mode, multi_sensor, iq_dir);
	SAMPLE_COMM_ISP_Run(0);

	// rkmpi init
	if (RK_MPI_SYS_Init() != RK_SUCCESS) {
		RK_LOGE("rk mpi sys init fail!");
		return -1;
	}

    // rtsp init	
	g_rtsplive = create_rtsp_demo(554);
	g_rtsp_session = rtsp_new_session(g_rtsplive, "/live/0");
	rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H264, NULL, 0);
	rtsp_sync_video_ts(g_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime());

	// vi init
	vi_dev_init();
	vi_chn_init(0, width, height);
	vi_chn_init(1, width, height);
	
	// venc init
	RK_CODEC_ID_E enCodecType = RK_VIDEO_ID_AVC;
	venc_init(0, width, height, enCodecType);

	// bind vi to venc	
	stSrcChn.enModId = RK_ID_VI;
	stSrcChn.s32DevId = 0;
	stSrcChn.s32ChnId = 0;
		
	stvencChn.enModId = RK_ID_VENC;
	stvencChn.s32DevId = 0;
	stvencChn.s32ChnId = 0;
	printf("====RK_MPI_SYS_Bind vi0 to venc0====\n");
	s32Ret = RK_MPI_SYS_Bind(&stSrcChn, &stvencChn);
	if (s32Ret != RK_SUCCESS) {
		RK_LOGE("bind 1 ch venc failed");
		return -1;
	}
			
	printf("init success\n");	
	
	pthread_t main_thread;
	pthread_create(&main_thread, NULL, GetMediaBuffer, NULL);
	pthread_t retina_thread;
	pthread_create(&retina_thread, NULL, RetinaProcessBuffer, NULL);
	
	while (1) {		
		usleep(50000);
	}

	pthread_join(main_thread, NULL);
	pthread_join(retina_thread, NULL);

	RK_MPI_SYS_UnBind(&stSrcChn, &stvencChn);
	RK_MPI_VI_DisableChn(0, 0);
	RK_MPI_VI_DisableChn(0, 1);
	
	RK_MPI_VENC_StopRecvFrame(0);
	RK_MPI_VENC_DestroyChn(0);
	
	RK_MPI_VI_DisableDev(0);

	RK_MPI_SYS_Exit();

	// Stop RKAIQ
	SAMPLE_COMM_ISP_Stop(0);

    // Release session
    HFReleaseInspireFaceSession(g_face_session);
    HFTerminateInspireFace();

	return 0;
}