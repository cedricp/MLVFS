#include "aces.h"
#include "aces_container/aces_Writer.h"
#include <libraw/libraw.h>
#include <OpenEXR/half.h>
#include "resource_manager.h"
#include "aces_idt/dng_idt.h"

static void mulVectorArray ( uint16_t * data_in,
                     half * data_out,
                     const uint32_t total,
                     const uint8_t dim,
                     const vector < vector < double > > & vct ) {
    
    for(uint32_t i = 0; i < total; i+=dim ) {
        double data_hin0 = (double)data_in[i] * (1.0/65535.0);
        double data_hin1 = (double)data_in[i+1] * (1.0/65535.0);
        double data_hin2 = (double)data_in[i+2] * (1.0/65535.0);

        data_out[i] = vct[0][0]*data_hin0 + vct[0][1]*data_hin1
        + vct[0][2]*data_hin2;
        data_out[i+1] = vct[1][0]*data_hin0 + vct[1][1]*data_hin1
        + vct[1][2]*data_hin2;
        data_out[i+2] = vct[2][0]*data_hin0 + vct[2][1]*data_hin1
        + vct[2][2]*data_hin2;
    }
}

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

extern "C" void process_aces(struct frame_headers * frame_headers, struct image_buffer* image_buffer, const char* name)
{
    LibRaw* raw_processor = new LibRaw;
    libraw_processed_image_t* image = NULL;
    int err;

   	err = raw_processor->open_buffer(image_buffer->header, image_buffer->header_size + image_buffer->size);
	if (err != LIBRAW_SUCCESS){
		printf("Libraw open buffer error\n");
        return;
	}
    
    if(raw_processor->unpack() != LIBRAW_SUCCESS){
		printf("Libraw unpack error\n");
        return;
	}

    	// XYZ colorspace
	raw_processor->imgdata.params.use_auto_wb = 0;
	raw_processor->imgdata.params.output_color = 5;
	raw_processor->imgdata.params.output_bps = 16;
	raw_processor->imgdata.params.gamm[0] = 1.0;
	raw_processor->imgdata.params.gamm[1] = 1.0;
	raw_processor->imgdata.params.no_interpolation= 0;
	raw_processor->imgdata.params.user_qual = 3; // See debayer method
	raw_processor->imgdata.params.use_camera_matrix = 1; // Use DNG matrix
	raw_processor->imgdata.params.use_camera_wb = 1;
	//raw_processor->imgdata.params.adjust_maximum_thr = _max_threshold;
	//raw_processor->imgdata.params.highlight = _highlight_mode;

	err = raw_processor->dcraw_process();
	if(err!= LIBRAW_SUCCESS){
		printf("dcraw process image error\n");
		return;
	}

    image = raw_processor->dcraw_make_mem_image(&err);
	if (err != LIBRAW_SUCCESS){
		printf("make mem image error\n");
		return;
	}

    free(image_buffer->header);
    image_buffer->header = NULL;
    
    vector < vector < double > > idt_matrix;
	DNGIdt dngidt = DNGIdt(raw_processor->imgdata.rawdata);
	idt_matrix = dngidt.getDNGIDTMatrix();

    // Highlight processing here

    size_t pixel_count = frame_headers->rawi_hdr.xRes * frame_headers->rawi_hdr.yRes * 3;
    uint16_t* in_buffer = (uint16_t*)image->data;
    half* out_buffer = (half*)malloc(pixel_count * sizeof(half));
    
    mulVectorArray(in_buffer, out_buffer, pixel_count, 3, idt_matrix);
    
    vector < std::string > filenames;
    filenames.push_back("/home/cedric/Desktop/test.exr");//name);

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

    for (int i=0;i < frame_headers->rawi_hdr.yRes; ++i){
        half* rgbData = out_buffer + frame_headers->rawi_hdr.xRes * 3 * i;
        writer.storeHalfRow ((halfBytes*)rgbData, i);
    }

    free(out_buffer);
    image_buffer->header_size = 0;
    image_buffer->size = writer.getOutputFileSize();
    image_buffer->header = NULL;
    image_buffer->data = (uint16_t*)writer.getExrBuffer();

    delete raw_processor;
}
