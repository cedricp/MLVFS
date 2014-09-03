/*
 * Copyright (C) 2014 David Milligan
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the
 * Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/param.h>
#include <string.h>
#include "raw.h"
#include "mlv.h"
#include "dng.h"

#include "dng_tag_codes.h"
#include "dng_tag_types.h"
#include "dng_tag_values.h"

#define IFD0_COUNT 33
#define EXIF_IFD_COUNT 5
#define MLVFS_SOFTWARE_NAME "MLVFS"
#define PACK(a) (((uint16_t)a[1] << 16) | ((uint16_t)a[0]))
#define PACK2(a,b) (((uint16_t)b << 16) | ((uint16_t)a))
#define STRING_ENTRY(a,b,c) ttAscii, (strlen(a) + 1), add_string(a, b, c)
#define RATIONAL_ENTRY(a,b,c,d) (d/2), add_array(a, b, c, d)
#define RATIONAL_ENTRY2(a,b,c,d) 1, add_rational(a, b, c, d)
#define ARRAY_ENTRY(a,b,c,d) d, add_array(a, b, c, d)
#define LINEARIZATION_TABLE(a,b,c) a, add_linearization_table(a, b, c)
#define HEADER_SIZE 65536

static uint16_t tiff_header[] = { byteOrderII, magicTIFF, 8, 0};

static int32_t daylight_wbal[] = {473635,1000000,1000000,1000000,624000,1000000};

struct directory_entry {
    uint16_t tag;
    uint16_t type;
    uint32_t count;
    uint32_t value;
};

//CDNG tag codes
enum
{
	tcTimeCodes				= 51043,
    tcFrameRate             = 51044,
    tcTStop                 = 51058,
    tcReelName              = 51081,
    tcCameraLabel           = 51105,
};


static uint32_t add_linearization_table(uint32_t max, uint8_t * buffer, size_t * data_offset)
{
    uint32_t result = *data_offset;
    for(uint16_t curr = 0; curr < max; curr++)
    {
        *(uint16_t*)(buffer + *data_offset) = curr;
        *data_offset += sizeof(uint16_t);
    }
    return result;
}

static uint32_t add_array(int32_t * array, uint8_t * buffer, size_t * data_offset, size_t length)
{
    uint32_t result = *data_offset;
    memcpy(buffer + result, array, length * sizeof(int32_t));
    *data_offset += length * sizeof(int32_t);
    return result;
}

static uint32_t add_string(char * str, uint8_t * buffer, size_t * data_offset)
{
    uint32_t result = 0;
    size_t length = strlen(str) + 1;
    if(length <= 4)
    {
        //we can fit in 4 bytes, so just pack the string into result
        memcpy(&result, str, length);
    }
    else
    {
        result = *data_offset;
        memcpy(buffer + result, str, length);
        *data_offset += length;
        //align to 2 bytes
        if(*data_offset % 2) *data_offset += 1;
    }
    return result;
}

static uint32_t add_rational(int32_t numerator, int32_t denominator, uint8_t * buffer, size_t * data_offset)
{
    uint32_t result = *data_offset;
    *(int32_t*)(buffer + *data_offset) = numerator;
    *data_offset += sizeof(int32_t);
    *(int32_t*)(buffer + *data_offset) = denominator;
    *data_offset += sizeof(int32_t);
    return result;
}

static void add_ifd(struct directory_entry * ifd, uint8_t * header, size_t * position, int count)
{
    *(uint16_t*)(header + *position) = count;
    *position += sizeof(uint16_t);
    memcpy(header + *position, ifd, count * sizeof(struct directory_entry));
    *position += count * sizeof(struct directory_entry);
    *(uint32_t*)(header + *position) = 0;
    *position += sizeof(uint32_t);
}

size_t dng_get_header_data(struct frame_headers * frame_headers, uint8_t * output_buffer, off_t offset, size_t max_size)
{
    /*
    - build the tiff header in a buffer
    - then copy the buffer to the output buffer according to offset and max_size
    this shouldn't be a big performance hit and it's a lot easier than trying
    to only generate the requested section of the header (most of the time the
    entire header will be requested all at once anyway, since typically the 
    requested size is at least 64kB)
    */
    size_t header_size = dng_get_header_size(frame_headers);
    uint8_t * header = malloc(header_size);
    size_t position = 0;
    if(header)
    {
        memset(header, 0 , header_size);
        memcpy(header + position, tiff_header, sizeof(tiff_header));
        position += sizeof(tiff_header);
        char make[32];
        char * model = (char*)frame_headers->idnt_hdr.cameraName;
        if(!model) model = "???";
        //make is usually the first word of cameraName
        strncpy(make, model, 32);
        char * space = strchr(make, ' ');
        if(space) *space = 0x0;
        
        size_t exif_ifd_offset = position + sizeof(uint16_t) + IFD0_COUNT * sizeof(struct directory_entry) + sizeof(uint32_t);
        size_t data_offset = exif_ifd_offset + sizeof(uint16_t) + EXIF_IFD_COUNT * sizeof(struct directory_entry) + sizeof(uint32_t);
        
        int bpp = frame_headers->rawi_hdr.raw_info.bits_per_pixel;
        //we get the active area of the original raw source, not the recorded data, so overwrite the active area if the recorded data does
        //not contain the OB areas
        if(frame_headers->rawi_hdr.xRes < frame_headers->rawi_hdr.raw_info.active_area.x1 + frame_headers->rawi_hdr.raw_info.active_area.x2 ||
           frame_headers->rawi_hdr.yRes < frame_headers->rawi_hdr.raw_info.active_area.y1 + frame_headers->rawi_hdr.raw_info.active_area.y2)
        {
            frame_headers->rawi_hdr.raw_info.active_area.x1 = 0;
            frame_headers->rawi_hdr.raw_info.active_area.y1 = 0;
            frame_headers->rawi_hdr.raw_info.active_area.x2 = frame_headers->rawi_hdr.xRes;
            frame_headers->rawi_hdr.raw_info.active_area.y2 = frame_headers->rawi_hdr.yRes;
        }
        int32_t frame_rate[2] = {frame_headers->file_hdr.sourceFpsNom, frame_headers->file_hdr.sourceFpsDenom};
        if(frame_rate[0] % 1000 == 976 && frame_rate[1] == 1000)
        {
            frame_rate[0] += 24;
            frame_rate[1]++;
        }
        
        struct directory_entry IFD0[IFD0_COUNT] =
        {
            {tcNewSubFileType,              ttLong,     1,      sfMainImage},
            {tcImageWidth,                  ttLong,     1,      frame_headers->rawi_hdr.xRes},
            {tcImageLength,                 ttLong,     1,      frame_headers->rawi_hdr.yRes},
            {tcBitsPerSample,               ttShort,    1,      16},
            {tcCompression,                 ttShort,    1,      ccUncompressed},
            {tcPhotometricInterpretation,   ttShort,    1,      piCFA},
            {tcFillOrder,                   ttShort,    1,      1},
            {tcMake,                        STRING_ENTRY(make, header, &data_offset)},
            {tcModel,                       STRING_ENTRY(model, header, &data_offset)},
            {tcStripOffsets,                ttLong,     1,      header_size},
            {tcOrientation,                 ttShort,    1,      1},
            {tcSamplesPerPixel,             ttShort,    1,      1},
            {tcRowsPerStrip,                ttShort,    1,      frame_headers->rawi_hdr.yRes},
            {tcStripByteCounts,             ttLong,     1,      dng_get_image_size(frame_headers)},
            {tcPlanarConfiguration,         ttShort,    1,      pcInterleaved},
            {tcSoftware,                    STRING_ENTRY(MLVFS_SOFTWARE_NAME, header, &data_offset)},
            {tcDateTime,                    STRING_ENTRY("", header, &data_offset)}, //TODO: implement
            {tcCFARepeatPatternDim,         ttShort,    2,      0x00020002}, //2x2
            {tcCFAPattern,                  ttByte,     4,      0x02010100}, //RGGB
            {tcExifIFD,                     ttLong,     1,      exif_ifd_offset},
            {tcDNGVersion,                  ttByte,     4,      0x00000401}, //1.4.0.0 in little endian
            {tcUniqueCameraModel,           STRING_ENTRY(model, header, &data_offset)},
            {tcLinearizationTable,          ttShort,    LINEARIZATION_TABLE((1 << bpp) - 1, header, &data_offset)},
            {tcBlackLevel,                  ttLong,     1,      frame_headers->rawi_hdr.raw_info.black_level},
            {tcWhiteLevel,                  ttLong,     1,      frame_headers->rawi_hdr.raw_info.white_level},
            {tcDefaultCropOrigin,           ttShort,    2,      PACK(frame_headers->rawi_hdr.raw_info.crop.origin)},
            {tcDefaultCropSize,             ttShort,    2,      PACK2(frame_headers->rawi_hdr.xRes,frame_headers->rawi_hdr.yRes)},
            {tcColorMatrix1,                ttSRational,RATIONAL_ENTRY(frame_headers->rawi_hdr.raw_info.color_matrix1, header, &data_offset, 18)},
            {tcAsShotNeutral,               ttRational, RATIONAL_ENTRY(daylight_wbal, header, &data_offset, 6)}, //TODO: get actual wbal
            {tcBaselineExposure,            ttSRational,RATIONAL_ENTRY(frame_headers->rawi_hdr.raw_info.exposure_bias, header, &data_offset, 2)},
            {tcActiveArea,                  ttLong,     ARRAY_ENTRY(frame_headers->rawi_hdr.raw_info.dng_active_area, header, &data_offset, 4)},
            {tcFrameRate,                   ttSRational,RATIONAL_ENTRY(frame_rate, header, &data_offset, 2)},
            {tcBaselineExposureOffset,      ttSRational,RATIONAL_ENTRY2(0, 1, header, &data_offset)},
        };
        
        struct directory_entry EXIF_IFD[EXIF_IFD_COUNT] =
        {
            {tcExposureTime,                ttRational, RATIONAL_ENTRY2(1, 30, header, &data_offset)}, //TODO: implement
            {tcFNumber,                     ttRational, RATIONAL_ENTRY2(frame_headers->lens_hdr.aperture, 100, header, &data_offset)},
            {tcISOSpeedRatings,             ttShort,    1,      frame_headers->expo_hdr.isoValue},
            {tcSensitivityType,             ttShort,    1,      stISOSpeed},
            {tcExifVersion,                 ttUndefined,4,      0x30333230},
        };
        
        add_ifd(IFD0, header, &position, IFD0_COUNT);
        add_ifd(EXIF_IFD, header, &position, EXIF_IFD_COUNT);
        
        size_t output_size = MIN(max_size, header_size - (size_t)MIN(0, offset));
        if(output_size)
        {
            memcpy(output_buffer, header + offset, output_size);
        }
        free(header);
        return output_size;
    }
    return 0;
}

size_t dng_get_header_size(struct frame_headers * frame_headers)
{
    return HEADER_SIZE;
}

/**
 * 14-bit encoding:
 
 hi          lo
 aaaaaaaa aaaaaabb
 bbbbbbbb bbbbcccc
 cccccccc ccdddddd
 dddddddd eeeeeeee
 eeeeeeff ffffffff
 ffffgggg gggggggg
 gghhhhhh hhhhhhhh
 */
size_t dng_get_image_data(struct frame_headers * frame_headers, FILE * file, uint8_t * output_buffer, off_t offset, size_t max_size)
{
    //unpack bits to 16 bit little endian (LSB first)
    int bpp = frame_headers->rawi_hdr.raw_info.bits_per_pixel;
    uint64_t pixel_start_index = (size_t)MAX(0, offset) / 2; //lets hope offsets are always even for now
    uint64_t pixel_start_address = pixel_start_index * bpp / 16;
    size_t output_size = max_size - (offset < 0 ? (size_t)(-offset) : 0);
    uint64_t pixel_count = output_size / 2;
    uint64_t packed_size = (pixel_count + 2) * bpp / 16;
    uint16_t * packed_bits = malloc((size_t)(packed_size * 2));
    uint8_t * buffer = output_buffer + (offset < 0 ? (size_t)(-offset) : 0) + offset % 2;
    if(packed_bits)
    {
        fseeko(file, frame_headers->position + frame_headers->vidf_hdr.frameSpace + sizeof(mlv_vidf_hdr_t) + pixel_start_address * 2, SEEK_SET);
        if(fread(packed_bits, (size_t)packed_size * 2, 1, file))
        {
            uint32_t mask = (1 << bpp) - 1;
            for(size_t pixel_index = 0; pixel_index < pixel_count; pixel_index++)
            {
                uint64_t pixel_address = (pixel_index + pixel_start_index) * bpp / 16 - pixel_start_address;
                uint64_t pixel_offset = (pixel_index + pixel_start_index) * bpp % 16;
                uint32_t data = ((packed_bits[pixel_address] << 16) & 0xFFFF0000) | (packed_bits[pixel_address + 1] & 0xFFFF);
                *(uint16_t*)(buffer + pixel_index * 2) = (uint16_t)((data >> ((32 - bpp) - pixel_offset)) & mask);
            }
        }
        else
        {
            fprintf(stderr, "Error reading source data");
        }
        free(packed_bits);
    }
    return max_size;
}

size_t dng_get_image_size(struct frame_headers * frame_headers)
{
    return (frame_headers->vidf_hdr.blockSize - frame_headers->vidf_hdr.frameSpace - sizeof(mlv_vidf_hdr_t)) * 16 / frame_headers->rawi_hdr.raw_info.bits_per_pixel; //16 bit
}

size_t dng_get_size(struct frame_headers * frame_headers)
{
    return dng_get_header_size(frame_headers) + dng_get_image_size(frame_headers);
}
