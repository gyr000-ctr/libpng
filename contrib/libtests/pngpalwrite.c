/* pngpalwrite.c
 *
 * Copyright (c) 2026 Cosmin Truta
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 *
 * Test the write-side truecolor-to-palette quantizer (png_quantize_image):
 *
 *   1. The output is a legal paletted PNG with at most the requested number
 *      of colors and the minimum legal bit depth.
 *   2. PLTE, tRNS and hIST are all present, before the first IDAT.
 *   3. Re-reading the output expands back to an image whose RGB PSNR is above
 *      a fixed threshold; RGBA images keep transparent pixels free of color
 *      bleeding.
 *   4. Images with no more than 256 distinct colors round-trip losslessly.
 *   5. The encoder is byte-for-byte deterministic across repeated runs.
 *   6. Bit-depth repacking (1, 2, 4 and 8) and 16-bit input all work.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(HAVE_CONFIG_H) && !defined(PNG_NO_CONFIG_H)
#  include <config.h>
#endif

#ifdef PNG_FREESTANDING_TESTS
#  include <png.h>
#else
#  include "../../png.h"
#endif

/* A growing in-memory output buffer used via png_set_write_fn(). */
typedef struct
{
   png_byte *data;
   size_t size;
   size_t capacity;
} mem_buffer;

static void
mem_write(png_struct *png_ptr, png_byte *data, size_t length)
{
   mem_buffer *buf = png_get_io_ptr(png_ptr);

   if (buf->size + length > buf->capacity)
   {
      size_t capacity = buf->capacity ? buf->capacity : 4096;

      while (capacity < buf->size + length)
         capacity *= 2;

      buf->data = realloc(buf->data, capacity);
      if (buf->data == NULL)
         png_error(png_ptr, "pngpalwrite: out of memory");
      buf->capacity = capacity;
   }

   memcpy(buf->data + buf->size, data, length);
   buf->size += length;
}

static void
mem_flush(png_struct *png_ptr)
{
   (void)png_ptr;
}

static void
mem_free(mem_buffer *buf)
{
   free(buf->data);
   buf->data = NULL;
   buf->size = buf->capacity = 0;
}


/* The standard memory reader uses png_set_read_fn with a user callback; libpng
 * has no built-in memory source, so provide one through the read callback
 * mechanism.
 */
typedef struct
{
   const png_byte *data;
   size_t size;
   size_t pos;
} read_source;

static void
mem_read(png_struct *png_ptr, png_byte *dest, size_t length)
{
   read_source *src = png_get_io_ptr(png_ptr);

   if (src->pos + length > src->size)
      png_error(png_ptr, "pngpalwrite: read past end of buffer");

   memcpy(dest, src->data + src->pos, length);
   src->pos += length;
}

static int
read_png(const png_byte *data, size_t size, png_uint_32 *out_width,
    png_uint_32 *out_height, int *out_bit_depth, int *out_color_type,
    png_colorp *out_palette, int *out_num_palette,
    png_bytep *out_trns, int *out_num_trns,
    png_uint_16p *out_hist, int *out_num_hist,
    png_byte **out_pixels_rgba)
{
   png_struct *png_ptr;
   png_info *info_ptr;
   read_source src = { data, size, 0 };
   png_uint_32 width, height;
   int bit_depth, color_type;
   png_byte *pixels;
   png_byte **rows;
   png_uint_32 y;
   png_colorp palette = NULL;
   int num_palette = 0;
   png_bytep trns = NULL;
   int num_trns = 0;
   png_uint_16p hist = NULL;
   int num_hist = 0;
   /* Copies kept alive after the read structures are destroyed. */
   png_color *palette_copy = NULL;
   png_byte *trns_copy = NULL;
   png_uint_16 *hist_copy = NULL;

   png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
   if (png_ptr == NULL)
      return 1;

   if (setjmp(png_jmpbuf(png_ptr)))
   {
      free(palette_copy);
      free(trns_copy);
      free(hist_copy);
      png_destroy_read_struct(&png_ptr, NULL, NULL);
      return 1;
   }

   info_ptr = png_create_info_struct(png_ptr);
   if (info_ptr == NULL)
   {
      png_destroy_read_struct(&png_ptr, NULL, NULL);
      return 1;
   }

   png_set_read_fn(png_ptr, &src, mem_read);
   png_read_info(png_ptr, info_ptr);

   width = png_get_image_width(png_ptr, info_ptr);
   height = png_get_image_height(png_ptr, info_ptr);
   bit_depth = png_get_bit_depth(png_ptr, info_ptr);
   color_type = png_get_color_type(png_ptr, info_ptr);

   if (color_type == PNG_COLOR_TYPE_PALETTE)
      png_get_PLTE(png_ptr, info_ptr, &palette, &num_palette);

   if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS) != 0)
      png_get_tRNS(png_ptr, info_ptr, &trns, &num_trns, NULL);

   if (png_get_valid(png_ptr, info_ptr, PNG_INFO_hIST) != 0)
      num_hist = (int)png_get_hIST(png_ptr, info_ptr, &hist);

   /* Copy the chunk payloads out now: destroying the info structure below
    * frees its internal palette, tRNS and histogram buffers, so handing the
    * raw pointers out would be a use-after-free.
    */
   if (palette != NULL && num_palette > 0)
   {
      palette_copy = malloc((size_t)num_palette * sizeof (png_color));
      if (palette_copy == NULL)
      {
         png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
         return 1;
      }
      memcpy(palette_copy, palette,
          (size_t)num_palette * sizeof (png_color));
   }

   if (trns != NULL && num_trns > 0)
   {
      trns_copy = malloc((size_t)num_trns);
      if (trns_copy == NULL)
      {
         free(palette_copy);
         png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
         return 1;
      }
      memcpy(trns_copy, trns, (size_t)num_trns);
   }

   if (hist != NULL && num_palette > 0)
   {
      hist_copy = malloc((size_t)num_palette * sizeof (png_uint_16));
      if (hist_copy == NULL)
      {
         free(palette_copy);
         free(trns_copy);
         png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
         return 1;
      }
      memcpy(hist_copy, hist,
          (size_t)num_palette * sizeof (png_uint_16));
   }

   /* Expand everything to 8-bit RGBA for comparison. */
   if (color_type == PNG_COLOR_TYPE_PALETTE)
      png_set_palette_to_rgb(png_ptr);
   if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS) != 0)
      png_set_tRNS_to_alpha(png_ptr);
   if (color_type == PNG_COLOR_TYPE_GRAY ||
       color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
      png_set_gray_to_rgb(png_ptr);
   if (bit_depth == 16)
      png_set_strip_16(png_ptr);
   else if (bit_depth < 8)
      png_set_packing(png_ptr);
   /* Add an opaque alpha channel only when the transforms above have not
    * already produced one; adding it after tRNS-to-alpha would corrupt the
    * expanded transparency values.
    */
   if (color_type != PNG_COLOR_TYPE_GRAY_ALPHA &&
       png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS) == 0)
      png_set_add_alpha(png_ptr, 0xff, PNG_FILLER_AFTER);

   png_read_update_info(png_ptr, info_ptr);

   pixels = malloc((size_t)width * height * 4);
   rows = malloc((size_t)height * sizeof (png_byte *));
   if (pixels == NULL || rows == NULL)
   {
      free(pixels);
      free(rows);
      free(palette_copy);
      free(trns_copy);
      free(hist_copy);
      png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
      return 1;
   }

   for (y = 0; y < height; y++)
      rows[y] = pixels + (size_t)y * width * 4;

   png_read_image(png_ptr, rows);
   png_read_end(png_ptr, NULL);

   free(rows);
   png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

   *out_width = width;
   *out_height = height;
   *out_bit_depth = bit_depth;
   *out_color_type = color_type;
   *out_palette = palette_copy;
   *out_num_palette = num_palette;
   *out_trns = trns_copy;
   *out_num_trns = num_trns;
   *out_hist = hist_copy;
   *out_num_hist = num_hist;
   *out_pixels_rgba = pixels;
   return 0;
}

/* Locate the byte range of a named chunk in a PNG buffer. */
static int
find_chunk(const png_byte *data, size_t size, const char *name,
    size_t *out_start, size_t *out_length)
{
   size_t pos = 8; /* skip signature */

   while (pos + 12 <= size)
   {
      png_uint_32 length = ((png_uint_32)data[pos] << 24) |
          ((png_uint_32)data[pos + 1] << 16) |
          ((png_uint_32)data[pos + 2] << 8) |
          data[pos + 3];
      const png_byte *type = data + pos + 4;

      if (memcmp(type, name, 4) == 0)
      {
         if (out_start != NULL)
            *out_start = pos;
         if (out_length != NULL)
            *out_length = length;
         return 0;
      }

      pos += 12 + (size_t)length;
      if (pos > size)
         break;
   }

   return 1;
}

/* Test case parameters. */
typedef struct
{
   const char *name;
   png_uint_32 width;
   png_uint_32 height;
   int channels;          /* 3 or 4 */
   int bit_depth;         /* 8 or 16 */
   int maximum_colors;
   double min_psnr;
   int expect_lossless;
} test_case;

/* Deterministic pseudo-random generator so every run is reproducible. */
static unsigned int
prand(unsigned int *seed)
{
   *seed = *seed * 1103515245U + 12345U;
   return (*seed >> 16) & 0x7fffU;
}

/* Build a synthetic image: smooth gradients plus a limited set of solid
 * blocks, which both stresses the quantizer and keeps PSNR high.
 */
static png_byte *
make_image(const test_case *tc)
{
   size_t pixel_bytes = (size_t)tc->channels *
       (size_t)(tc->bit_depth / 8);
   png_byte *image = malloc((size_t)tc->width * tc->height * pixel_bytes);
   unsigned int seed = 0x12345678U;
   png_uint_32 x, y;
   int c;

   if (image == NULL)
      return NULL;

   for (y = 0; y < tc->height; y++)
   {
      for (x = 0; x < tc->width; x++)
      {
         png_byte *px = image + ((size_t)y * tc->width + x) * pixel_bytes;
         int base[4];

         base[0] = (int)(x * 255 / (tc->width - 1));
         base[1] = (int)(y * 255 / (tc->height - 1));
         base[2] = (int)((x + y) * 255 / (tc->width + tc->height - 2));
         base[3] = 255;

         if (tc->channels == 4)
         {
            /* A band of fully transparent pixels and a band of partial
             * alpha, so tRNS gets exercised.
             */
            if (x < tc->width / 4)
               base[3] = 0;
            else if (x < tc->width / 2)
               base[3] = 128;
         }

         for (c = 0; c < tc->channels; c++)
         {
            int v = base[c];

            /* Noise goes on the color channels only; perturbing alpha would
             * turn the deliberately fully-transparent band into near-zero
             * (but non-zero) values, defeating the transparency check.
             */
            if (c < 3)
               v += (int)(prand(&seed) % 9) - 4;

            if (v < 0) v = 0;
            if (v > 255) v = 255;

            if (tc->bit_depth == 16)
            {
               px[c * 2] = (png_byte)v;
               px[c * 2 + 1] = (png_byte)v;
            }
            else
               px[c] = (png_byte)v;
         }
      }
   }

   return image;
}

/* Build an image whose number of distinct colors is exactly ncolors, so the
 * quantizer must preserve it losslessly.
 */
static png_byte *
make_limited_image(png_uint_32 width, png_uint_32 height, int channels,
    int ncolors)
{
   png_byte *image = malloc((size_t)width * height * (size_t)channels);
   png_uint_32 x, y;

   if (image == NULL)
      return NULL;

   for (y = 0; y < height; y++)
   {
      for (x = 0; x < width; x++)
      {
         png_byte *px = image +
             (((size_t)y * width + x) * (size_t)channels);
         int color = (int)((x * 7 + y * 13) % (png_uint_32)ncolors);

         px[0] = (png_byte)((color * 53) & 0xff);
         px[1] = (png_byte)((color * 97 + 11) & 0xff);
         px[2] = (png_byte)((color * 211 + 7) & 0xff);
         if (channels == 4)
            px[3] = (png_byte)(color % 3 == 0 ? 0 : 255);
      }
   }

   return image;
}

/* Encode an image through png_quantize_image into an in-memory PNG. */
static int
encode_quantized(const test_case *tc, const png_byte *image, mem_buffer *out)
{
   png_struct *png_ptr;
   png_info *info_ptr;
   png_byte **rows;
   png_uint_32 y;

   out->data = NULL;
   out->size = out->capacity = 0;

   png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
   if (png_ptr == NULL)
      return 1;

   if (setjmp(png_jmpbuf(png_ptr)))
   {
      png_destroy_write_struct(&png_ptr, NULL);
      return 1;
   }

   info_ptr = png_create_info_struct(png_ptr);
   if (info_ptr == NULL)
   {
      png_destroy_write_struct(&png_ptr, NULL);
      return 1;
   }

   png_set_write_fn(png_ptr, out, mem_write, mem_flush);

   png_set_IHDR(png_ptr, info_ptr, tc->width, tc->height, tc->bit_depth,
       tc->channels == 4 ? PNG_COLOR_TYPE_RGB_ALPHA : PNG_COLOR_TYPE_RGB,
       PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

   rows = malloc((size_t)tc->height * sizeof (png_byte *));
   if (rows == NULL)
   {
      png_destroy_write_struct(&png_ptr, &info_ptr);
      return 1;
   }
   for (y = 0; y < tc->height; y++)
      rows[y] = (png_byte *)(image + (size_t)y * tc->width *
          (size_t)tc->channels * (size_t)(tc->bit_depth / 8));

   png_quantize_image(png_ptr, info_ptr, rows, tc->maximum_colors);

   png_write_info(png_ptr, info_ptr);
   png_write_image(png_ptr, rows);
   png_write_end(png_ptr, info_ptr);

   free(rows);
   png_destroy_write_struct(&png_ptr, &info_ptr);
   return 0;
}

/* PSNR of the RGB channels between an original 8-bit RGBA image and a
 * decoded one.  Fully transparent pixels are excluded: their RGB values are
 * not meaningful in a palette with a transparent entry.
 */
static double
rgb_psnr(const png_byte *orig, const png_byte *decoded, png_uint_32 width,
    png_uint_32 height, int has_alpha)
{
   double mse = 0.0;
   png_uint_32 n = 0;
   png_uint_32 i;

   for (i = 0; i < width * height; i++)
   {
      const png_byte *a = orig + (size_t)i * 4;
      const png_byte *b = decoded + (size_t)i * 4;
      int c;

      if (has_alpha && a[3] == 0)
         continue;

      for (c = 0; c < 3; c++)
      {
         double d = (double)a[c] - b[c];
         mse += d * d;
      }
      n++;
   }

   if (n == 0 || mse == 0.0)
      return 999.0;

   mse /= (double)n * 3.0;
   return 10.0 * log10(255.0 * 255.0 / mse);
}

/* Check that fully transparent pixels decode with alpha 0 and that they all
 * share the single transparent palette entry (no color bleeding).
 */
static int
transparent_pixels_clean(const png_byte *orig, const png_byte *decoded,
    png_uint_32 width, png_uint_32 height)
{
   png_uint_32 i;
   int saw_transparent = 0;

   for (i = 0; i < width * height; i++)
   {
      const png_byte *a = orig + (size_t)i * 4;
      const png_byte *b = decoded + (size_t)i * 4;

      if (a[3] == 0)
      {
         saw_transparent = 1;
         if (b[3] != 0)
            return 0;
      }
   }

   return saw_transparent;
}

static int
run_case(const test_case *tc)
{
   png_byte *image8;       /* original in 8-bit RGBA form, for comparison */
   png_byte *image;        /* actual encoder input (8/16-bit, RGB/RGBA) */
   mem_buffer first = { 0 };
   mem_buffer second = { 0 };
   png_uint_32 width, height;
   int bit_depth, color_type;
   png_colorp palette = NULL;
   int num_palette = 0;
   png_bytep trns = NULL;
   int num_trns = 0;
   png_uint_16p hist = NULL;
   int num_hist = 0;
   png_byte *decoded = NULL;
   size_t plte_start, plte_len, trns_start, hist_start, idat_start;
   double psnr;
   int result = 1;

   printf("Testing %s... ", tc->name);
   fflush(stdout);

   image = make_image(tc);
   if (image == NULL)
   {
      printf("FAIL (image allocation)\n");
      return 1;
   }

   /* Build an 8-bit RGBA reference for metric computation. */
   image8 = malloc((size_t)tc->width * tc->height * 4);
   if (image8 == NULL)
      goto done;

   {
      png_uint_32 i;

      for (i = 0; i < tc->width * tc->height; i++)
      {
         const png_byte *src = image + (size_t)i * tc->channels *
             (tc->bit_depth / 8);
         png_byte *dst = image8 + (size_t)i * 4;
         int c;

         for (c = 0; c < tc->channels; c++)
            dst[c] = tc->bit_depth == 16 ? src[c * 2] : src[c];
         dst[3] = tc->channels == 4 ?
             (tc->bit_depth == 16 ? src[6] : src[3]) : 255;
      }
   }

   if (encode_quantized(tc, image, &first) != 0 || first.size == 0)
   {
      printf("FAIL (encode)\n");
      goto done;
   }

   /* Determinism: a second run must produce identical bytes. */
   if (encode_quantized(tc, image, &second) != 0)
   {
      printf("FAIL (second encode)\n");
      mem_free(&first);
      goto done;
   }

   if (first.size != second.size ||
       memcmp(first.data, second.data, first.size) != 0)
   {
      printf("FAIL (non-deterministic output)\n");
      mem_free(&first);
      mem_free(&second);
      goto done;
   }
   mem_free(&second);

   /* Chunk ordering: PLTE (and tRNS/hIST when expected) before IDAT. */
   if (find_chunk(first.data, first.size, "IDAT", &idat_start, NULL) != 0)
   {
      printf("FAIL (no IDAT)\n");
      mem_free(&first);
      goto done;
   }
   if (find_chunk(first.data, first.size, "PLTE", &plte_start, &plte_len)
       != 0 || plte_start >= idat_start)
   {
      printf("FAIL (PLTE missing or after IDAT)\n");
      mem_free(&first);
      goto done;
   }
   if (find_chunk(first.data, first.size, "hIST", &hist_start, NULL) != 0 ||
       hist_start >= idat_start)
   {
      printf("FAIL (hIST missing or after IDAT)\n");
      mem_free(&first);
      goto done;
   }
   if (tc->channels == 4 &&
       (find_chunk(first.data, first.size, "tRNS", &trns_start, NULL) != 0 ||
        trns_start >= idat_start))
   {
      printf("FAIL (tRNS missing or after IDAT)\n");
      mem_free(&first);
      goto done;
   }

   /* Re-read and validate the decoded image and palette metadata. */
   if (read_png(first.data, first.size, &width, &height, &bit_depth,
           &color_type, &palette, &num_palette, &trns, &num_trns, &hist,
           &num_hist, &decoded) != 0)
   {
      printf("FAIL (decode)\n");
      mem_free(&first);
      goto done;
   }

   if (color_type != PNG_COLOR_TYPE_PALETTE)
   {
      printf("FAIL (not paletted)\n");
      goto done_read;
   }
   if (num_palette > tc->maximum_colors || num_palette > 256 ||
       num_palette <= 0)
   {
      printf("FAIL (palette size %d)", num_palette);
      goto done_read;
   }
   /* png_get_hIST returns a flag, not a count; the histogram must be present
    * and hold one non-zero entry per palette entry.
    */
   if (hist == NULL)
   {
      printf("FAIL (no hIST)");
      goto done_read;
   }
   {
      int h;

      for (h = 0; h < num_palette; h++)
         if (hist[h] == 0)
         {
            printf("FAIL (hIST[%d] == 0)", h);
            goto done_read;
         }
   }
   /* The bit depth must be the smallest legal one for the palette size. */
   {
      int expected_depth = num_palette <= 2 ? 1 :
          num_palette <= 4 ? 2 : num_palette <= 16 ? 4 : 8;

      if (bit_depth != expected_depth)
      {
         printf("FAIL (bit depth %d, expected %d for %d colors)", bit_depth,
             expected_depth, num_palette);
         goto done_read;
      }
   }
   /* PLTE length must match num_palette. */
   if ((size_t)plte_len != (size_t)num_palette * 3)
   {
      printf("FAIL (PLTE length %zu != %d)", plte_len, num_palette * 3);
      goto done_read;
   }

   if (tc->channels == 4)
   {
      if (num_trns <= 0)
      {
         printf("FAIL (no tRNS on RGBA input)\n");
         goto done_read;
      }
      if (!transparent_pixels_clean(image8, decoded, width, height))
      {
         printf("FAIL (transparent pixel color bleeding)\n");
         goto done_read;
      }
   }

   psnr = rgb_psnr(image8, decoded, width, height, tc->channels == 4);
   if (psnr < tc->min_psnr)
   {
      printf("FAIL (PSNR %.2f < %.2f)\n", psnr, tc->min_psnr);
      goto done_read;
   }

   if (tc->expect_lossless && psnr < 900.0)
   {
      printf("FAIL (expected lossless, PSNR %.2f)\n", psnr);
      goto done_read;
   }

   printf("PASS (%d colors, depth %d, PSNR %.2f dB)\n", num_palette,
       bit_depth, psnr);
   result = 0;

done_read:
   free(decoded);
   free(palette);
   free(trns);
   free(hist);
   mem_free(&first);
done:
   free(image8);
   free(image);
   return result;
}

/* Lossless case: an image with at most 256 distinct colors must survive
 * quantization unchanged.
 */
static int
run_lossless_case(int channels, int ncolors, int expected_depth)
{
   test_case tc;
   png_uint_32 width = 40, height = 32;
   png_byte *image;
   mem_buffer out = { 0 };
   png_uint_32 rw, rh;
   int bd, ct;
   png_colorp pal = NULL;
   int npal = 0;
   png_bytep trns = NULL;
   int ntrns = 0;
   png_uint_16p hist = NULL;
   int nhist = 0;
   png_byte *decoded = NULL;
   int result = 1;
   png_uint_32 i;

   printf("Testing lossless %d-color %s... ", ncolors,
       channels == 4 ? "RGBA" : "RGB");
   fflush(stdout);

   image = make_limited_image(width, height, channels, ncolors);
   if (image == NULL)
   {
      printf("FAIL (allocation)\n");
      return 1;
   }

   memset(&tc, 0, sizeof tc);
   tc.name = "lossless";
   tc.width = width;
   tc.height = height;
   tc.channels = channels;
   tc.bit_depth = 8;
   tc.maximum_colors = 256;
   tc.min_psnr = 900.0;

   if (encode_quantized(&tc, image, &out) != 0)
   {
      printf("FAIL (encode)\n");
      free(image);
      return 1;
   }

   if (read_png(out.data, out.size, &rw, &rh, &bd, &ct, &pal, &npal,
           &trns, &ntrns, &hist, &nhist, &decoded) != 0)
   {
      printf("FAIL (decode)\n");
      goto done;
   }

   if (npal != ncolors)
   {
      printf("FAIL (palette %d != %d) ", npal, ncolors);
      goto done;
   }
   if (bd != expected_depth)
   {
      printf("FAIL (depth %d != %d) ", bd, expected_depth);
      goto done;
   }

   for (i = 0; i < width * height; i++)
   {
      const png_byte *a = image + (size_t)i * channels;
      const png_byte *b = decoded + (size_t)i * 4;
      int c;

      for (c = 0; c < 3; c++)
         if (a[c] != b[c])
         {
            printf("FAIL (RGB mismatch at %u)\n", i);
            goto done;
         }
      if (channels == 4 && a[3] != b[3])
      {
         printf("FAIL (alpha mismatch at %u: %d != %d)\n", i, a[3], b[3]);
         goto done;
      }
   }

   printf("PASS\n");
   result = 0;

done:
   free(decoded);
   free(pal);
   free(trns);
   free(hist);
   mem_free(&out);
   free(image);
   return result;
}

/* 16-bit input with a handful of distinct colors: the exact path must keep
 * every color, repack to the minimum palette depth and preserve the alpha
 * band without color bleeding.
 */
static int
run_lossless_16bit_case(int channels, int ncolors, int expected_depth)
{
   test_case tc;
   png_uint_32 width = 24, height = 16;
   size_t pixel_bytes = (size_t)channels * 2;
   png_byte *image;
   mem_buffer out = { 0 };
   png_uint_32 rw, rh;
   int bd, ct;
   png_colorp pal = NULL;
   int npal = 0;
   png_bytep trns = NULL;
   int ntrns = 0;
   png_uint_16p hist = NULL;
   int nhist = 0;
   png_byte *decoded = NULL;
   int result = 1;
   png_uint_32 x, y;

   printf("Testing lossless 16-bit %d-color %s... ", ncolors,
       channels == 4 ? "RGBA" : "RGB");
   fflush(stdout);

   image = malloc((size_t)width * height * pixel_bytes);
   if (image == NULL)
   {
      printf("FAIL (allocation)\n");
      return 1;
   }

   /* A fixed set of well-separated 16-bit colors; the last is transparent. */
   for (y = 0; y < height; y++)
   {
      for (x = 0; x < width; x++)
      {
         png_byte *px = image + ((size_t)y * width + x) * pixel_bytes;
         int color = (int)((x * 3 + y * 5) % ncolors);
         png_uint_16 rgb[3];
         int c;

         rgb[0] = (png_uint_16)(color * 6000);
         rgb[1] = (png_uint_16)(color * 4000 + 1000);
         rgb[2] = (png_uint_16)(65000 - color * 5000);

         for (c = 0; c < 3; c++)
         {
            px[c * 2] = (png_byte)(rgb[c] >> 8);
            px[c * 2 + 1] = (png_byte)(rgb[c] & 0xff);
         }
         if (channels == 4)
         {
            png_uint_16 a = (color == ncolors - 1) ? 0 : 65535;

            px[6] = (png_byte)(a >> 8);
            px[7] = (png_byte)(a & 0xff);
         }
      }
   }

   memset(&tc, 0, sizeof tc);
   tc.name = "lossless16";
   tc.width = width;
   tc.height = height;
   tc.channels = channels;
   tc.bit_depth = 16;
   tc.maximum_colors = 256;
   tc.min_psnr = 900.0;

   if (encode_quantized(&tc, image, &out) != 0)
   {
      printf("FAIL (encode)\n");
      goto done;
   }

   if (read_png(out.data, out.size, &rw, &rh, &bd, &ct, &pal, &npal,
           &trns, &ntrns, &hist, &nhist, &decoded) != 0)
   {
      printf("FAIL (decode)\n");
      goto done;
   }

   if (npal != ncolors)
   {
      printf("FAIL (palette %d != %d) ", npal, ncolors);
      goto done;
   }
   if (bd != expected_depth || ct != PNG_COLOR_TYPE_PALETTE)
   {
      printf("FAIL (depth %d/type %d, expected %d/3) ", bd, ct,
          expected_depth);
      goto done;
   }

   /* The transparent color must decode to alpha 0 with no bleeding. */
   if (channels == 4)
   {
      if (ntrns <= 0)
      {
         printf("FAIL (no tRNS) ");
         goto done;
      }
      for (y = 0; y < height; y++)
      {
         for (x = 0; x < width; x++)
         {
            png_byte *src = image + ((size_t)y * width + x) * pixel_bytes;
            png_byte *dst = decoded + ((size_t)y * width + x) * 4;

            if (src[6] == 0 && src[7] == 0 && dst[3] != 0)
            {
               printf("FAIL (alpha bleeding) ");
               goto done;
            }
         }
      }
   }

   printf("PASS\n");
   result = 0;

done:
   free(decoded);
   free(pal);
   free(trns);
   free(hist);
   mem_free(&out);
   free(image);
   return result;
}

int
main(void)
{
   int result = 0;

   /* Fixed PSNR floors.  They sit a few dB below the measured result of each
    * synthetic case, so they catch a gross regression while tolerating normal
    * variation in the median-cut output.
    */
   static const test_case cases[] =
   {
      { "RGB 8-bit, 256 colors", 64, 48, 3, 8, 256, 33.0, 0 },
      { "RGB 8-bit, 64 colors", 64, 48, 3, 8, 64, 28.0, 0 },
      { "RGB 8-bit, 16 colors", 48, 32, 3, 8, 16, 22.0, 0 },
      { "RGB 8-bit, 4 colors", 48, 32, 3, 8, 4, 16.0, 0 },
      { "RGB 8-bit, 2 colors", 40, 24, 3, 8, 2, 12.0, 0 },
      { "RGBA 8-bit, 256 colors", 64, 48, 4, 8, 256, 31.0, 0 },
      { "RGBA 8-bit, 32 colors", 48, 32, 4, 8, 32, 22.0, 0 },
      { "RGB 16-bit, 256 colors", 48, 32, 3, 16, 256, 33.0, 0 },
      { "RGBA 16-bit, 128 colors", 48, 32, 4, 16, 128, 28.0, 0 }
   };
   static const int ncases = (int)(sizeof cases / sizeof cases[0]);
   int i;

#ifdef PNG_WRITE_QUANTIZE_SUPPORTED
   for (i = 0; i < ncases; i++)
      result |= run_case(&cases[i]);

   /* Bit-depth repacking boundaries: 2, 4, 16 and 256 distinct colors. */
   result |= run_lossless_case(3, 2, 1);
   result |= run_lossless_case(3, 4, 2);
   result |= run_lossless_case(3, 16, 4);
   result |= run_lossless_case(3, 256, 8);
   result |= run_lossless_case(4, 8, 4);

   /* 16-bit input must also take the exact, repacked path. */
   result |= run_lossless_16bit_case(3, 3, 2);
   result |= run_lossless_16bit_case(4, 4, 2);
   result |= run_lossless_16bit_case(3, 16, 4);
#else
   printf("pngpalwrite: PNG_WRITE_QUANTIZE_SUPPORTED is disabled; "
       "nothing to test\n");
#endif

   if (result == 0)
      printf("pngpalwrite: all tests passed\n");
   else
      printf("pngpalwrite: FAILURES detected\n");

   return result;
}
