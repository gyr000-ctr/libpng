/* pngwquant.c - write-side truecolor to palette quantization
 *
 * Copyright (c) 2026 Cosmin Truta
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 *
 * This file implements a median-cut color quantizer used by the write code:
 * a whole RGB or RGBA image (8-bit or 16-bit per component) is reduced to a
 * palette of at most 256 entries, the IHDR is reconfigured for a paletted
 * output of the minimum legal bit depth, and the PLTE, tRNS and hIST chunks
 * are populated before the first IDAT is written.  png_do_write_quantize()
 * then maps every row to palette indices.
 *
 * The algorithm is deterministic: given the same input it always produces the
 * same palette, indices and output file.  Ties when selecting the split
 * channel and when searching for the nearest entry are broken by channel and
 * entry order, respectively.  No libc qsort is used; the median partition is
 * done by an in-place quickselect with a deterministic pivot.
 */

#include "pngpriv.h"

#ifdef PNG_WRITE_QUANTIZE_SUPPORTED

/* A box of the median-cut partition.  The pixel indices live in a single
 * shared array; each box refers to a contiguous run [first, first+count).
 */
typedef struct
{
   png_uint_32 first;
   png_uint_32 count;
   png_uint_16 lo[4];
   png_uint_16 hi[4];
} png_quantize_box;

/* Return a pointer to the pixel with the given global index. */
static const png_byte *
png_quantize_pixel(const png_struct *png_ptr, png_uint_32 idx)
{
   const png_byte *row =
       png_ptr->quantize_rows[idx / png_ptr->quantize_width];

   return row + (idx % png_ptr->quantize_width) *
       (size_t)png_ptr->quantize_channels *
       (png_ptr->quantize_bit_depth / 8);
}

/* Read one component (0: red, 1: green, 2: blue, 3: alpha) of a pixel. */
static png_uint_16
png_quantize_component(const png_struct *png_ptr, const png_byte *px, int c)
{
   size_t offset = (size_t)c * (png_ptr->quantize_bit_depth / 8);

   if (png_ptr->quantize_bit_depth == 16)
      return (png_uint_16)(((png_uint_32)px[offset] << 8) |
          px[offset + 1]);

   return px[offset];
}

/* Order two pixel indices by the current split channel, then by the full
 * component vector and finally by the index itself.  The total order makes
 * the partition stable and independent of any library sorting routine.
 */
static int
png_quantize_less(const png_struct *png_ptr, png_uint_32 ia, png_uint_32 ib)
{
   const png_byte *pa = png_quantize_pixel(png_ptr, ia);
   const png_byte *pb = png_quantize_pixel(png_ptr, ib);
   int c;
   int first = png_ptr->quantize_split_channel;

   for (c = 0; c < png_ptr->quantize_channels; c++)
   {
      int channel = (first + c) % png_ptr->quantize_channels;
      png_uint_32 va = png_quantize_component(png_ptr, pa, channel);
      png_uint_32 vb = png_quantize_component(png_ptr, pb, channel);

      if (va != vb)
         return va < vb;
   }

   return ia < ib;
}

static void
png_quantize_swap(png_uint_32 *a, png_uint_32 *b)
{
   png_uint_32 tmp = *a;
   *a = *b;
   *b = tmp;
}

/* Rearrange indices[lo..hi] so that the element that would sit at position k
 * in sorted order is placed there, smaller elements precede it and larger
 * elements follow it.  Deterministic pivot: the first element, with insertion
 * sort for short runs to avoid quadratic behavior on nearly-sorted data.
 */
static void
png_quantize_quickselect(png_struct *png_ptr, png_uint_32 *indices,
    png_uint_32 lo, png_uint_32 hi, png_uint_32 k)
{
   while (lo < hi)
   {
      png_uint_32 pivot, i, j, store;

      if (hi - lo < 8)
      {
         for (i = lo + 1; i <= hi; i++)
         {
            png_uint_32 value = indices[i];

            j = i;
            while (j > lo &&
                png_quantize_less(png_ptr, value, indices[j - 1]))
            {
               indices[j] = indices[j - 1];
               j--;
            }
            indices[j] = value;
         }
         return;
      }

      pivot = indices[lo];
      store = lo;

      png_quantize_swap(&indices[lo], &indices[hi]);
      for (i = lo; i < hi; i++)
      {
         if (png_quantize_less(png_ptr, indices[i], pivot))
         {
            png_quantize_swap(&indices[i], &indices[store]);
            store++;
         }
      }
      png_quantize_swap(&indices[store], &indices[hi]);

      if (k == store)
         return;

      if (k < store)
         hi = store - 1;
      else
         lo = store + 1;
   }
}

/* Refresh the component extrema of the given box. */
static void
png_quantize_box_bounds(const png_struct *png_ptr, png_quantize_box *box)
{
   png_uint_32 i;
   int c;

   for (c = 0; c < 4; c++)
   {
      box->lo[c] = 65535U;
      box->hi[c] = 0;
   }

   for (i = 0; i < box->count; i++)
   {
      const png_byte *px = png_quantize_pixel(png_ptr,
          png_ptr->quantize_indices[box->first + i]);

      for (c = 0; c < png_ptr->quantize_channels; c++)
      {
         png_uint_16 v = png_quantize_component(png_ptr, px, c);

         if (v < box->lo[c])
            box->lo[c] = v;
         if (v > box->hi[c])
            box->hi[c] = v;
      }
   }
}

/* Scale a 16-bit component average down to an 8-bit palette component. */
static png_byte
png_quantize_to_8(png_uint_32 sum, png_uint_32 count, int bit_depth)
{
   png_uint_32 avg = (sum + count / 2) / count;

   if (bit_depth == 16)
      avg = (avg + 128U) / 257U;

   return (png_byte)avg;
}

/* Forward declaration: shared PLTE/tRNS/hIST allocation. */
static void png_quantize_alloc_output(png_struct *png_ptr);

/* Build the palette entries, transparency and histogram from the final
 * boxes.
 */
static void
png_quantize_make_palette(png_struct *png_ptr, const png_quantize_box *boxes,
    int num_boxes)
{
   png_uint_32 *sums;
   int b, c;
   png_uint_16 entries = (png_uint_16)num_boxes;

   sums = png_voidcast(png_uint_32 *,
       png_malloc(png_ptr, (png_alloc_size_t)4 * entries *
           sizeof (png_uint_32)));

   png_quantize_alloc_output(png_ptr);

   for (b = 0; b < num_boxes; b++)
   {
      png_uint_32 i;

      for (c = 0; c < 4; c++)
         sums[(png_alloc_size_t)b * 4 + (png_alloc_size_t)c] = 0;

      for (i = 0; i < boxes[b].count; i++)
      {
         const png_byte *px = png_quantize_pixel(png_ptr,
             png_ptr->quantize_indices[boxes[b].first + i]);

         for (c = 0; c < png_ptr->quantize_channels; c++)
            sums[(png_alloc_size_t)b * 4 + (png_alloc_size_t)c] +=
                png_quantize_component(png_ptr, px, c);
      }
   }

   /* Each box becomes one palette entry: the component average of its
    * pixels.  Averaging the alpha channel (rather than dropping it) keeps
    * semi-transparent colors separate from opaque ones, so transparent
    * pixels never inherit the RGB values of unrelated opaque colors.
    */
   for (b = 0; b < num_boxes; b++)
   {
      png_uint_32 count = boxes[b].count;

      png_ptr->quantize_palette[b].red =
          png_quantize_to_8(sums[(png_alloc_size_t)b * 4 + 0], count,
              png_ptr->quantize_bit_depth);
      png_ptr->quantize_palette[b].green =
          png_quantize_to_8(sums[(png_alloc_size_t)b * 4 + 1], count,
              png_ptr->quantize_bit_depth);
      png_ptr->quantize_palette[b].blue =
          png_quantize_to_8(sums[(png_alloc_size_t)b * 4 + 2], count,
              png_ptr->quantize_bit_depth);

      if (png_ptr->quantize_channels == 4)
         png_ptr->quantize_alpha[b] =
             png_quantize_to_8(sums[(png_alloc_size_t)b * 4 + 3], count,
                 png_ptr->quantize_bit_depth);
   }

   png_free(png_ptr, sums);

   /* hIST: relative frequency of every entry, scaled to the 16-bit range.
    * Rounding preserves non-zero counts so rare colors stay visible.
    */
   {
      png_uint_32 total = png_ptr->quantize_width * png_ptr->quantize_height;

      for (b = 0; b < num_boxes; b++)
      {
         png_uint_32 h = (png_uint_32)
             (((png_alloc_size_t)boxes[b].count * 65535U + total / 2) /
                 total);

         if (h > 65535U)
            h = 65535U;
         if (h == 0)
            h = 1;
         png_ptr->quantize_hist[b] = (png_uint_16)h;
      }
   }

   /* tRNS only needs entries up to the last non-opaque one. */
   if (png_ptr->quantize_channels == 4)
   {
      int num_trans = 0;

      for (b = 0; b < num_boxes; b++)
         if (png_ptr->quantize_alpha[b] != 255)
            num_trans = b + 1;

      png_ptr->quantize_num_trans = (png_uint_16)num_trans;
   }

   png_ptr->quantize_num_palette = entries;
}

/* Distance from an 8-bit pixel (one byte per channel) to a palette entry.
 * Used by the row mapper (16-bit input rows are reduced to 8 bits first).
 * The alpha term is weighted above a single color channel so that a fully
 * transparent pixel can never be matched to an opaque palette entry and vice
 * versa ("color bleeding").
 */
static png_uint_32
png_quantize_distance_8(const png_struct *png_ptr, const png_byte *px,
    int entry)
{
   png_uint_32 dr, dg, db, da;
   const png_color *pal = &png_ptr->quantize_palette[entry];

   dr = (png_uint_32)px[0] - pal->red;
   dg = (png_uint_32)px[1] - pal->green;
   db = (png_uint_32)px[2] - pal->blue;
   da = png_ptr->quantize_channels == 4 ?
       (png_uint_32)px[3] - png_ptr->quantize_alpha[entry] : 0;

   return dr * dr + dg * dg + db * db + 4U * da * da;
}

/* Allocate the output PLTE/tRNS/hIST arrays. */
static void
png_quantize_alloc_output(png_struct *png_ptr)
{
   png_ptr->quantize_palette = png_voidcast(png_color *,
       png_calloc(png_ptr, PNG_MAX_PALETTE_LENGTH * sizeof (png_color)));
   png_ptr->quantize_alpha = png_voidcast(png_byte *,
       png_malloc(png_ptr, PNG_MAX_PALETTE_LENGTH));
   png_ptr->quantize_hist = png_voidcast(png_uint_16 *,
       png_calloc(png_ptr, PNG_MAX_PALETTE_LENGTH *
           sizeof (png_uint_16)));
   memset(png_ptr->quantize_alpha, 0xff, PNG_MAX_PALETTE_LENGTH);
}

/* A distinct-color key holds the original (input-depth) RGBA components.
 * RGB inputs store 65535 in the alpha slot.  Keeping the four 16-bit words
 * avoids any reliance on a 64-bit integer type.
 */
typedef struct
{
   png_uint_16 component[4];
} png_quantize_key;

/* Deterministic 32-bit hash of a four-component key. */
static png_uint_32
png_quantize_hash64(const png_quantize_key *key)
{
   png_uint_32 h = 2166136261U;
   int c;

   for (c = 0; c < 4; c++)
   {
      h ^= (png_uint_32)key->component[c] & 0xffU;
      h *= 16777619U;
      h ^= (png_uint_32)key->component[c] >> 8;
      h *= 16777619U;
   }

   return h;
}

static int
png_quantize_key_equal(const png_quantize_key *a, const png_quantize_key *b)
{
   int c;

   for (c = 0; c < 4; c++)
      if (a->component[c] != b->component[c])
         return 0;

   return 1;
}

/* Exact-color fast path for 8-bit and 16-bit input.  If the image contains
 * no more than maximum_colors distinct colors, build a palette containing
 * every distinct color (so the result is lossless at the 8-bit output
 * precision) and an exact histogram, then return 1.  Return 0 when the colors
 * exceed the limit, so the caller falls back to median cut.  Deterministic:
 * entries are numbered in first-appearance order and the hash scan is linear.
 */
static int
png_quantize_exact(png_struct *png_ptr, png_uint_32 npixels, int maximum_colors)
{
   int channels = png_ptr->quantize_channels;
   int bit_depth = png_ptr->quantize_bit_depth;
   png_uint_32 capacity;
   png_quantize_key *keys;
   png_uint_16 *slot_entry;
   png_byte *occupied;
   png_uint_32 *counts;
   png_uint_32 i;
   int num_palette = 0;
   int num_trans = 0;

   /* Use a power-of-two table sized to at least twice the pixel count, capped
    * so a pathological image cannot request an unbounded table.
    */
   capacity = 256;
   while (capacity < npixels * 2 && capacity < (1U << 22))
      capacity *= 2;
   if (npixels * 2 > capacity)
      return 0; /* table would be too dense; fall back to median cut */

   keys = png_voidcast(png_quantize_key *,
       png_malloc_warn(png_ptr, (png_alloc_size_t)capacity *
           sizeof (png_quantize_key)));
   slot_entry = png_voidcast(png_uint_16 *,
       png_malloc_warn(png_ptr, (png_alloc_size_t)capacity *
           sizeof (png_uint_16)));
   occupied = png_voidcast(png_byte *,
       png_malloc_warn(png_ptr, capacity));
   counts = png_voidcast(png_uint_32 *,
       png_malloc_warn(png_ptr,
           (png_alloc_size_t)PNG_MAX_PALETTE_LENGTH *
           sizeof (png_uint_32)));

   if (keys == NULL || slot_entry == NULL || occupied == NULL ||
       counts == NULL)
   {
      png_free(png_ptr, keys);
      png_free(png_ptr, slot_entry);
      png_free(png_ptr, occupied);
      png_free(png_ptr, counts);
      return 0;
   }

   memset(occupied, 0, capacity);
   memset(counts, 0,
       (png_alloc_size_t)PNG_MAX_PALETTE_LENGTH * sizeof (png_uint_32));

   for (i = 0; i < npixels; i++)
   {
      const png_byte *px = png_quantize_pixel(png_ptr, i);
      png_quantize_key key;
      png_uint_32 slot;
      int c, entry;

      for (c = 0; c < channels; c++)
         key.component[c] = png_quantize_component(png_ptr, px, c);
      key.component[3] = channels == 4 ?
          png_quantize_component(png_ptr, px, 3) : 65535U;

      slot = png_quantize_hash64(&key) & (capacity - 1);

      while (occupied[slot] && !png_quantize_key_equal(&keys[slot], &key))
         slot = (slot + 1) & (capacity - 1);

      if (occupied[slot])
      {
         entry = slot_entry[slot];
      }
      else
      {
         if (num_palette >= maximum_colors)
         {
            png_free(png_ptr, keys);
            png_free(png_ptr, slot_entry);
            png_free(png_ptr, occupied);
            png_free(png_ptr, counts);
            return 0; /* too many distinct colors */
         }

         entry = num_palette++;
         occupied[slot] = 1;
         keys[slot] = key;
         slot_entry[slot] = (png_uint_16)entry;
      }

      counts[entry]++;
   }

   png_quantize_alloc_output(png_ptr);

   /* Walk the image again in pixel order; the first pixel of each distinct
    * color defines that entry's RGB and alpha, giving a stable
    * first-appearance ordering.  Every entry is assigned at least once.  The
    * 16-bit components are reduced to the 8-bit precision the PLTE stores.
    */
   for (i = 0; i < npixels; i++)
   {
      const png_byte *px = png_quantize_pixel(png_ptr, i);
      png_quantize_key key;
      png_uint_32 slot;
      int c, entry;

      for (c = 0; c < channels; c++)
         key.component[c] = png_quantize_component(png_ptr, px, c);
      key.component[3] = channels == 4 ?
          png_quantize_component(png_ptr, px, 3) : 65535U;

      slot = png_quantize_hash64(&key) & (capacity - 1);
      while (!png_quantize_key_equal(&keys[slot], &key))
         slot = (slot + 1) & (capacity - 1);
      entry = slot_entry[slot];

      png_ptr->quantize_palette[entry].red =
          png_quantize_to_8(key.component[0], 1, bit_depth);
      png_ptr->quantize_palette[entry].green =
          png_quantize_to_8(key.component[1], 1, bit_depth);
      png_ptr->quantize_palette[entry].blue =
          png_quantize_to_8(key.component[2], 1, bit_depth);
      if (channels == 4)
         png_ptr->quantize_alpha[entry] =
             png_quantize_to_8(key.component[3], 1, bit_depth);
   }

   /* Determine the tRNS extent and the histogram. */
   for (i = 0; i < (png_uint_32)num_palette; i++)
   {
      if (channels == 4 && png_ptr->quantize_alpha[i] != 255)
         num_trans = (int)i + 1;

      {
         png_uint_32 h = (png_uint_32)
             (((png_alloc_size_t)counts[i] * 65535U + npixels / 2) /
                 npixels);

         if (h == 0)
            h = 1;
         png_ptr->quantize_hist[i] = (png_uint_16)h;
      }
   }

   png_free(png_ptr, keys);
   png_free(png_ptr, slot_entry);
   png_free(png_ptr, occupied);
   png_free(png_ptr, counts);

   png_ptr->quantize_num_palette = (png_uint_16)num_palette;
   png_ptr->quantize_num_trans = (png_uint_16)num_trans;
   return 1;
}

/* Apply the finished palette to the info structure and enable the row
 * transformation.  Shared by the exact and median-cut paths.
 */
static void
png_quantize_apply(png_struct *png_ptr, png_info *info_ptr, png_uint_32 width)
{
   png_byte out_bit_depth;

   if (png_ptr->quantize_num_palette <= 2)
      out_bit_depth = 1;
   else if (png_ptr->quantize_num_palette <= 4)
      out_bit_depth = 2;
   else if (png_ptr->quantize_num_palette <= 16)
      out_bit_depth = 4;
   else
      out_bit_depth = 8;

   /* png_write_info() -> png_write_IHDR() copies these into the png_struct
    * when the IHDR chunk is emitted.
    */
   info_ptr->bit_depth = out_bit_depth;
   info_ptr->color_type = PNG_COLOR_TYPE_PALETTE;
   info_ptr->channels = 1;
   info_ptr->pixel_depth = out_bit_depth;
   info_ptr->rowbytes = PNG_ROWBYTES(out_bit_depth, width);

   /* PLTE, tRNS and hIST are all written before the first IDAT. */
   png_set_PLTE(png_ptr, info_ptr, png_ptr->quantize_palette,
       (int)png_ptr->quantize_num_palette);

   if (png_ptr->quantize_num_trans > 0)
      png_set_tRNS(png_ptr, info_ptr, png_ptr->quantize_alpha,
          (int)png_ptr->quantize_num_trans, NULL);

   png_set_hIST(png_ptr, info_ptr, png_ptr->quantize_hist);

   /* Rows become one index byte each; PNG_PACK compacts them in place when
    * the output depth is below 8.  The flags survive png_write_IHDR(); the
    * user row geometry is restored in png_quantize_start_row().
    */
   png_ptr->transformations |= PNG_WRITE_QUANTIZE;
   if (out_bit_depth < 8)
      png_ptr->transformations |= PNG_PACK;
}

void /* PRIVATE */
png_quantize_build(png_struct *png_ptr, png_info *info_ptr, png_bytepp image,
    int maximum_colors)
{
   png_uint_32 width, height, npixels;
   png_quantize_box *boxes;
   int num_boxes;
   int b, c;
   int max_colors;
   png_byte input_color_type = info_ptr->color_type;
   png_byte input_bit_depth = info_ptr->bit_depth;

   png_debug1(1, "in png_quantize_build (maximum_colors=%d)", maximum_colors);

   if (png_ptr == NULL || info_ptr == NULL || image == NULL)
      png_error(png_ptr, "png_quantize_build: invalid argument");

   /* At this point png_set_IHDR() has populated the info structure only; the
    * png_struct IHDR fields are filled in by png_write_IHDR() during
    * png_write_info().
    */
   if (input_color_type != PNG_COLOR_TYPE_RGB &&
       input_color_type != PNG_COLOR_TYPE_RGB_ALPHA)
      png_error(png_ptr,
          "png_quantize_build: input must be truecolor RGB or RGBA");

   if (input_bit_depth != 8 && input_bit_depth != 16)
      png_error(png_ptr, "png_quantize_build: input bit depth must be 8 or 16");

   if ((png_ptr->mode & PNG_WROTE_INFO_BEFORE_PLTE) != 0)
      png_error(png_ptr,
          "png_quantize_build: must be called before png_write_info");

   if (maximum_colors <= 0 || maximum_colors > PNG_MAX_PALETTE_LENGTH)
      max_colors = PNG_MAX_PALETTE_LENGTH;
   else
      max_colors = maximum_colors;

   width = info_ptr->width;
   height = info_ptr->height;
   npixels = width * height;

   if (npixels == 0)
      png_error(png_ptr, "png_quantize_build: empty image");

   png_ptr->quantize_channels =
       input_color_type == PNG_COLOR_TYPE_RGB ? (png_byte)3 : (png_byte)4;
   png_ptr->quantize_bit_depth = input_bit_depth;
   png_ptr->quantize_width = width;
   png_ptr->quantize_height = height;
   png_ptr->quantize_rows = image;
   png_ptr->quantize_num_palette = 0;
   png_ptr->quantize_num_trans = 0;
   png_ptr->quantize_split_channel = 0;

   /* Prefer the exact, lossless palette when the color count fits. */
   if (png_quantize_exact(png_ptr, npixels, max_colors))
   {
      png_quantize_apply(png_ptr, info_ptr, width);
      return;
   }

   /* Median-cut fallback for images with more than maximum_colors colors. */
   png_ptr->quantize_indices = png_voidcast(png_uint_32 *,
       png_malloc(png_ptr, (png_alloc_size_t)npixels *
           sizeof (png_uint_32)));

   {
      png_uint_32 i;

      for (i = 0; i < npixels; i++)
         png_ptr->quantize_indices[i] = i;
   }

   /* One box covers the whole image initially. */
   boxes = png_voidcast(png_quantize_box *,
       png_malloc(png_ptr, (png_alloc_size_t)max_colors *
           sizeof (png_quantize_box)));
   num_boxes = 1;
   boxes[0].first = 0;
   boxes[0].count = npixels;
   png_quantize_box_bounds(png_ptr, &boxes[0]);

   /* Median cut: repeatedly split the box with the widest component range. */
   while (num_boxes < max_colors)
   {
      int split = -1;
      int split_channel = 0;
      png_uint_32 split_range = 0;

      for (b = 0; b < num_boxes; b++)
      {
         for (c = 0; c < png_ptr->quantize_channels; c++)
         {
            png_uint_32 range = (png_uint_32)boxes[b].hi[c] -
                boxes[b].lo[c];

            /* Only multi-pixel, non-constant boxes qualify.  On a range
             * tie the lower channel, then the earlier box, wins; this keeps
             * the partition deterministic.
             */
            if (boxes[b].count > 1 && range > split_range)
            {
               split_range = range;
               split = b;
               split_channel = c;
            }
         }
      }

      if (split < 0)
         break; /* every remaining box holds one distinct color */

      {
         png_quantize_box *box = &boxes[split];
         png_uint_32 median = box->count / 2;

         /* Partition the box's run around its median on the split channel. */
         png_ptr->quantize_split_channel = (png_byte)split_channel;
         png_quantize_quickselect(png_ptr, png_ptr->quantize_indices,
             box->first, box->first + box->count - 1,
             box->first + median);

         /* Split into two non-empty halves. */
         boxes[num_boxes].first = box->first + median;
         boxes[num_boxes].count = box->count - median;
         box->count = median;
         num_boxes++;

         png_quantize_box_bounds(png_ptr, box);
         png_quantize_box_bounds(png_ptr, &boxes[num_boxes - 1]);
      }
   }

   png_quantize_make_palette(png_ptr, boxes, num_boxes);

   /* The median-cut work area is no longer needed; row mapping uses the
    * palette and the original image rows.
    */
   png_free(png_ptr, png_ptr->quantize_indices);
   png_ptr->quantize_indices = NULL;
   png_free(png_ptr, boxes);

   png_quantize_apply(png_ptr, info_ptr, width);
}

/* Called from png_write_start_row() before the row buffer is allocated:
 * restore the truecolor user row geometry so that the incoming RGB/RGBA rows
 * fit, and so png_do_write_quantize() sees the correct pixel layout.
 */
void /* PRIVATE */
png_quantize_start_row(png_struct *png_ptr)
{
   if ((png_ptr->transformations & PNG_WRITE_QUANTIZE) == 0)
      return;

   png_ptr->usr_channels = png_ptr->quantize_channels;
   png_ptr->usr_bit_depth = png_ptr->quantize_bit_depth;
}

void /* PRIVATE */
png_do_write_quantize(png_struct *png_ptr, png_row_info *row_info,
    png_byte *row)
{
   png_uint_32 x;
   int entries = (int)png_ptr->quantize_num_palette;
   png_uint_32 channels = png_ptr->quantize_channels;
   int input_depth = png_ptr->quantize_bit_depth;
   /* After the optional 16-to-8 reduction below, mapping reads 8-bit
    * pixels of one byte per channel.
    */
   size_t pixel_size = (size_t)channels;

   png_debug(1, "in png_do_write_quantize");

   /* A 16-bit truecolor row cannot be overwritten byte-by-byte in place (each
    * input pixel is bigger than the one output index byte).  Reduce every
    * component to 8 bits first, packing toward the start of the row; the
    * palette stores 8-bit components, so no information is lost relative to
    * what the output can represent.  The persistent quantizer state is left
    * untouched, so every row is reduced independently.
    */
   if (input_depth == 16)
   {
      for (x = 0; x < row_info->width; x++)
      {
         const png_byte *src = row + x * channels * 2;
         png_byte *dst = row + x * channels;
         png_uint_32 c;

         for (c = 0; c < channels; c++)
            dst[c] = src[c * 2]; /* take the high byte */
      }
   }

   /* Every input pixel becomes one palette index byte; PNG_PACK compacts the
    * indices in place afterwards when the output bit depth is below 8.
    */
   for (x = 0; x < row_info->width; x++)
   {
      const png_byte *px = row + x * pixel_size;
      png_uint_32 best_distance = 0xffffffffU;
      int best = 0;
      int e;

      for (e = 0; e < entries; e++)
      {
         png_uint_32 d = png_quantize_distance_8(png_ptr, px, e);

         /* Strict comparison keeps the first (lowest-index) nearest entry,
          * so identical distances map identically on every platform.
          */
         if (d < best_distance)
         {
            best_distance = d;
            best = e;
         }
      }

      row[x] = (png_byte)best;
   }

   row_info->color_type = PNG_COLOR_TYPE_PALETTE;
   row_info->channels = 1;
   row_info->bit_depth = 8;
   row_info->pixel_depth = 8;
   row_info->rowbytes = PNG_ROWBYTES(8, row_info->width);
}

#else /* PNG_WRITE_QUANTIZE_SUPPORTED */
/* Empty translation unit when the feature is disabled. */
#endif /* WRITE_QUANTIZE */
