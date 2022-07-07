#include "aces.h"
#include "aces_container/aces_Writer.h"
#include <libraw/libraw.h>
#include <OpenEXR/half.h>
#include "resource_manager.h"
#include "aces_idt/dng_idt.h"
#include <algorithm>

static void mulVectorArray ( uint16_t * data_in,
                     half * data_out,
                     const uint32_t total,
                     const uint8_t dim,
                     const vector < vector < double > > & vct,
                     double scale) {
    #pragma omp parallel for
    for(uint32_t i = 0; i < total; i+=dim ) {
        double data_hin0 = (double)data_in[i] * (1.0/65535.0) * scale;
        double data_hin1 = (double)data_in[i+1] * (1.0/65535.0) * scale;
        double data_hin2 = (double)data_in[i+2] * (1.0/65535.0) * scale;

        data_out[i] = vct[0][0]*data_hin0 + vct[0][1]*data_hin1
        + vct[0][2]*data_hin2;
        data_out[i+1] = vct[1][0]*data_hin0 + vct[1][1]*data_hin1
        + vct[1][2]*data_hin2;
        data_out[i+2] = vct[2][0]*data_hin0 + vct[2][1]*data_hin1
        + vct[2][2]*data_hin2;
    }
}

/*
    Need to find a smarter way to get file size
*/
extern "C" size_t exr_get_size(struct frame_headers * frame_headers, const char* name)
{
    vector < std::string > filenames;
    filenames.push_back(name);

    aces_Writer writer;

    MetaWriteClip writeParams;
    writeParams.duration				= 1;
    writeParams.outputFilenames			= filenames;
    writeParams.outputRows				= frame_headers->rawi_hdr.yRes;
    writeParams.outputCols				= frame_headers->rawi_hdr.xRes;
    writeParams.hi = writer.getDefaultHeaderInfo();
    writeParams.hi.originalImageFlag	= 1;

    writeParams.hi.channels.resize(3);
    writeParams.hi.channels[0].name = "B";
    writeParams.hi.channels[1].name = "G";
    writeParams.hi.channels[2].name = "R";

    DynamicMetadata dynamicMeta;
    dynamicMeta.imageIndex = 0;
    dynamicMeta.imageCounter = 0;

    writer.configure ( writeParams );
    writer.newImageObject ( dynamicMeta );

    return writer.getOutputFileSize();
}

/*
# - Debayer method
--+----------------------
0 - linear interpolation
1 - VNG interpolation
2 - PPG interpolation
3 - AHD interpolation
4 - DCB interpolation
11 - DHT intepolation
12 - Modified AHD intepolation (by Anton Petrusevich)
*/
extern "C" void process_aces(struct frame_headers * frame_headers, struct image_buffer* image_buffer, const char* name, struct mlvfs* mlvfs)
{
    LibRaw* raw_processor = new LibRaw;
    libraw_processed_image_t* image = NULL;
    int err = 0;

    err = raw_processor->open_buffer(image_buffer->header, image_buffer->header_size + image_buffer->size);
	if (err != LIBRAW_SUCCESS){
		err_printf("Libraw open buffer error\n");
        return;
	}
    
    err = raw_processor->unpack();
    if(err != LIBRAW_SUCCESS){
		err_printf("Libraw unpack error\n");
        return;
	}

	raw_processor->imgdata.params.use_auto_wb = 0;
	raw_processor->imgdata.params.output_color = 5; // XYZ
	raw_processor->imgdata.params.output_bps = 16;
	raw_processor->imgdata.params.gamm[0] = 1.0;
	raw_processor->imgdata.params.gamm[1] = 1.0;
	raw_processor->imgdata.params.no_interpolation= 0;
	raw_processor->imgdata.params.user_qual = mlvfs->debayer; // See debayer method
	raw_processor->imgdata.params.use_camera_matrix = 1; // Use DNG matrix
	raw_processor->imgdata.params.use_camera_wb = 1;
	raw_processor->imgdata.params.highlight = mlvfs->highlight;
	//raw_processor->imgdata.params.adjust_maximum_thr = _max_threshold;

    err = raw_processor->dcraw_process();
	if(err != LIBRAW_SUCCESS){
		err_printf("Dcraw process image error\n");
		return;
	}

    image = raw_processor->dcraw_make_mem_image(&err);
	if (err != LIBRAW_SUCCESS){
		err_printf("Dcraw make mem image error\n");
		return;
	}

    vector < vector < double > > idt_matrix;
	DNGIdt dngidt = DNGIdt(raw_processor->imgdata.rawdata);
	idt_matrix = dngidt.getDNGIDTMatrix();

    double wb_compensation = 1.0;
	if(mlvfs->highlight > 0){
		wb_compensation = ( *(std::max_element ( raw_processor->imgdata.color.pre_mul, raw_processor->imgdata.color.pre_mul+3)) /
						*(std::min_element ( raw_processor->imgdata.color.pre_mul, raw_processor->imgdata.color.pre_mul+3)) );
	}

    size_t pixel_count = frame_headers->rawi_hdr.xRes * frame_headers->rawi_hdr.yRes * 3;
    uint16_t* in_buffer = (uint16_t*)image->data;
    half* out_buffer = (half*)malloc(pixel_count * sizeof(half));
    
    mulVectorArray(in_buffer, out_buffer, pixel_count, 3, idt_matrix, mlvfs->headroom * wb_compensation);
    
    vector < std::string > filenames;
    filenames.push_back(name);

    aces_Writer writer;

    MetaWriteClip writeParams;
    writeParams.duration				= 1;
    writeParams.outputFilenames			= filenames;
    writeParams.outputRows				= frame_headers->rawi_hdr.yRes;
    writeParams.outputCols				= frame_headers->rawi_hdr.xRes;
    writeParams.hi = writer.getDefaultHeaderInfo();
    writeParams.hi.originalImageFlag	= 1;

    writeParams.hi.channels.resize(3);
    writeParams.hi.channels[0].name = "B";
    writeParams.hi.channels[1].name = "G";
    writeParams.hi.channels[2].name = "R";

    DynamicMetadata dynamicMeta;
    dynamicMeta.imageIndex = 0;
    dynamicMeta.imageCounter = 0;

    writer.configure ( writeParams );
    writer.newImageObject ( dynamicMeta );

    #pragma omp parallel for
    for (int i=0;i < frame_headers->rawi_hdr.yRes; ++i){
        half* rgbData = out_buffer + frame_headers->rawi_hdr.xRes * 3 * i;
        writer.storeHalfRow ((halfBytes*)rgbData, i);
    }

    free(out_buffer);
    free(image_buffer->header);
    image_buffer->header = NULL;
    image_buffer->header_size = 0;

    image_buffer->size = writer.getOutputFileSize();
    image_buffer->data = (uint16_t*)writer.getExrBuffer();

    delete raw_processor;
}
