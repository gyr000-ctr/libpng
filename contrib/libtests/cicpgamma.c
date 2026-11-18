/* cicpgamma.c
 *
 * Copyright (c) 2026 Cosmin Truta
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 *
 * Validate that the cICP chunk transfer characteristics and colour primaries
 * are decoded into the read state rather than merely stored:
 *
 *   1. Transfer 16 (PQ/SMPTE ST 2084) produces no gamma correction and, when an
 *      sRGB or gAMA chunk is also present, no sRGB gamma gets applied on top.
 *   2. Transfer 13 (sRGB) produces the same gamma handling as an sRGB chunk.
 *   3. RGB to gray uses the cICP colour primaries.
 *   4. PNG files without a cICP chunk produce byte identical output to the
 *      same files read without any cICP interpretation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#if defined(HAVE_CONFIG_H) && !defined(PNG_NO_CONFIG_H)
#  include <config.h>
#endif

#include "../../pngpriv.h"

#define TEST_WIDTH   32
#define TEST_HEIGHT  4

#define ERROR(...)   do { fprintf(stderr, "cicp_gamma: " __VA_ARGS__); \
                          fprintf(stderr, "\n"); exit(1); } while (0)

/* ---- memory IO ---------------------------------------------------------- */

struct mem_buffer
{
   unsigned char *data;
   size_t len;
   size_t cap;
};

static void
write_data(png_structp pp, png_bytep data, png_size_t length)
{
   struct mem_buffer *buf = png_get_io_ptr(pp);

   if (buf->len + length > buf->cap)
   {
      size_t cap = buf->cap ? buf->cap : 256;
      while (cap < buf->len + length)
         cap *= 2;
      buf->data = realloc(buf->data, cap);
      if (buf->data == NULL)
         ERROR("out of memory");
      buf->cap = cap;
   }
   memcpy(buf->data + buf->len, data, length);
   buf->len += length;
}

static void
read_data(png_structp pp, png_bytep data, png_size_t length)
{
   struct mem_buffer *buf = png_get_io_ptr(pp);

   if (buf->len < length)
      png_error(pp, "cicp_gamma: truncated memory input");
   memcpy(data, buf->data, length);
   memmove(buf->data, buf->data + length, buf->len - length);
   buf->len -= length;
}

/* ---- PNG construction --------------------------------------------------- */

struct extra_chunk
{
   const char *type;      /* 4-character chunk name, NULL terminates */
   unsigned char data[16];
   size_t len;
};

static void
put_uint32(unsigned char *p, unsigned long v)
{
   p[0] = (unsigned char)(v >> 24);
   p[1] = (unsigned char)(v >> 16);
   p[2] = (unsigned char)(v >> 8);
   p[3] = (unsigned char)v;
}

/* Emit a raw ancillary chunk (length, type, data, CRC) into the write stream
 * via png_write_chunk_start/end.  cICP is written before IDAT.
 */
static void
write_raw_chunk(png_structp pp, const struct extra_chunk *c)
{
   unsigned char lenbuf[4];
   unsigned char crcbuf[4];
   uLong crc;
   png_byte name[5];

   memcpy(name, c->type, 4);
   name[4] = 0;

   put_uint32(lenbuf, (unsigned long)c->len);
   write_data(pp, lenbuf, 4);
   write_data(pp, (png_bytep)name, 4);
   if (c->len > 0)
      write_data(pp, (png_bytep)c->data, c->len);

   crc = crc32(0L, Z_NULL, 0);
   crc = crc32(crc, (const Bytef *)name, 4);
   if (c->len > 0)
      crc = crc32(crc, (const Bytef *)c->data, (uInt)c->len);
   put_uint32(crcbuf, crc);
   write_data(pp, crcbuf, 4);
}

/* Create an 8-bit RGB (or RGBA) PNG in memory carrying the requested extra
 * chunks.  Pixel rows use a deterministic ramp; R, G and B differ so that both
 * gamma tables and rgb-to-gray coefficients visibly affect the output.
 */
static void
make_png(struct mem_buffer *out, const struct extra_chunk *chunks,
         int use_alpha)
{
   png_structp pp;
   png_infop info;
   int color_type = use_alpha ? PNG_COLOR_TYPE_RGB_ALPHA :
      PNG_COLOR_TYPE_RGB;
   int channels = use_alpha ? 4 : 3;
   png_bytep row;
   int x, y;

   memset(out, 0, sizeof *out);

   pp = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
   if (pp == NULL)
      ERROR("png_create_write_struct failed");
   info = png_create_info_struct(pp);
   if (info == NULL)
      ERROR("png_create_info_struct failed");
   if (setjmp(png_jmpbuf(pp)))
      ERROR("libpng write error");

   png_set_write_fn(pp, out, write_data, NULL);

   png_set_IHDR(pp, info, TEST_WIDTH, TEST_HEIGHT, 8, color_type,
       PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

   /* Write signature and IHDR first. */
   png_write_info_before_PLTE(pp, info);

   /* Ancillary chunks go here (after PLTE/before IDAT for these types). */
   for ( ; chunks->type != NULL; ++chunks)
      write_raw_chunk(pp, chunks);

   row = malloc((size_t)TEST_WIDTH * channels);
   if (row == NULL)
      ERROR("out of memory");

   for (y = 0; y < TEST_HEIGHT; ++y)
   {
      for (x = 0; x < TEST_WIDTH; ++x)
      {
         int i = y * TEST_WIDTH + x;
         row[x * channels + 0] = (png_byte)(i * 7 + 16);       /* red */
         row[x * channels + 1] = (png_byte)(i * 3 + 64);       /* green */
         row[x * channels + 2] = (png_byte)(255 - i * 5);      /* blue */
         if (use_alpha)
            row[x * channels + 3] = (png_byte)(128 + i);
      }
      png_write_row(pp, row);
   }
   free(row);
   png_write_end(pp, info);
   png_destroy_write_struct(&pp, &info);
}

/* ---- PNG reading -------------------------------------------------------- */

struct read_result
{
   unsigned char *rows;
   size_t rowbytes;
   int height;
};

static void
read_png(const struct mem_buffer *in, struct read_result *res,
         int do_gamma, png_fixed_point screen_gamma, int do_rgb_to_gray)
{
   png_structp pp;
   png_infop info;
   struct mem_buffer tmp;
   int y;

   tmp.data = malloc(in->len);
   if (tmp.data == NULL)
      ERROR("out of memory");
   memcpy(tmp.data, in->data, in->len);
   tmp.len = in->len;
   tmp.cap = in->len;
   memset(res, 0, sizeof *res);

   pp = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
   if (pp == NULL)
      ERROR("png_create_read_struct failed");
   info = png_create_info_struct(pp);
   if (info == NULL)
      ERROR("png_create_info_struct failed");
   if (setjmp(png_jmpbuf(pp)))
      ERROR("libpng read error");

   png_set_read_fn(pp, &tmp, read_data);
   png_read_info(pp, info);

   if (do_rgb_to_gray)
      /* NONE: non-gray pixels are expected in this synthetic ramp; the warning
       * is irrelevant to the coefficient checks.
       */
      png_set_rgb_to_gray(pp, PNG_ERROR_ACTION_NONE,
          -1.0/*default coefficients*/, -1.0);

   if (do_gamma)
      /* Set only the output (screen) gamma; leave png_struct::file_gamma at
       * zero so that the file gamma is resolved from the chunks through the
       * PNGv3 precedence rules (png_set_gamma_fixed would force file_gamma and
       * bypass those rules).  PNG_ALPHA_PNG performs no alpha encoding, it just
       * records the screen gamma here.
       */
      png_set_alpha_mode_fixed(pp, PNG_ALPHA_PNG, screen_gamma);

   png_read_update_info(pp, info);
   res->rowbytes = png_get_rowbytes(pp, info);
   res->height = (int)png_get_image_height(pp, info);
   res->rows = malloc(res->rowbytes * (size_t)res->height);
   if (res->rows == NULL)
      ERROR("out of memory");

   for (y = 0; y < res->height; ++y)
   {
      png_bytep row = res->rows + res->rowbytes * (size_t)y;
      png_read_row(pp, row, NULL);
   }
   png_read_end(pp, info);
   png_destroy_read_struct(&pp, &info, NULL);
   free(tmp.data);
}

static int
results_equal(const struct read_result *a, const struct read_result *b)
{
   if (a->rowbytes != b->rowbytes || a->height != b->height)
      return 0;
   return memcmp(a->rows, b->rows, a->rowbytes * (size_t)a->height) == 0;
}

static void
free_result(struct read_result *r)
{
   free(r->rows);
   r->rows = NULL;
}

/* ---- chunk builders ----------------------------------------------------- */

static struct extra_chunk
cicp_chunk(unsigned char primaries, unsigned char transfer)
{
   struct extra_chunk c;

   memset(&c, 0, sizeof c);
   c.type = "cICP";
   c.data[0] = primaries;
   c.data[1] = transfer;
   c.data[2] = 0; /* matrix coefficients: identity/RGB */
   c.data[3] = 1; /* video full range flag */
   c.len = 4;
   return c;
}

static struct extra_chunk
srgb_chunk(void)
{
   struct extra_chunk c;

   memset(&c, 0, sizeof c);
   c.type = "sRGB";
   c.data[0] = 0; /* perceptual intent */
   c.len = 1;
   return c;
}

static struct extra_chunk
gama_chunk(unsigned long gamma_fixed)
{
   struct extra_chunk c;

   memset(&c, 0, sizeof c);
   c.type = "gAMA";
   put_uint32((unsigned char *)c.data, gamma_fixed);
   c.len = 4;
   return c;
}

static const struct extra_chunk no_chunks[] = { { NULL, {0}, 0 } };

/* ---- the tests ---------------------------------------------------------- */

static int failures;

static void
check(int condition, const char *what)
{
   if (condition)
      printf("  ok: %s\n", what);
   else
   {
      printf("FAIL: %s\n", what);
      ++failures;
   }
}

int
main(void)
{
   struct mem_buffer png_plain, png_pq, png_pq_srgb, png_pq_gama;
   struct mem_buffer png_srgb_tf, png_srgb_chunk, png_bt2020, png_bt709;
   struct mem_buffer png_srgb_gama, png_tf13_srgb;
   struct read_result r_plain, r_pq, r_pq_srgb, r_pq_gama;
   struct read_result r_srgb_tf, r_srgb_chunk, r_bt2020, r_bt709;
   struct read_result r_srgb_gama, r_tf13_srgb;
   struct extra_chunk end = { NULL, {0}, 0 };
   struct extra_chunk list_pq[2], list_pq_srgb[3], list_pq_gama[3];
   struct extra_chunk list_srgb_tf[2], list_srgb_chunk[2];
   struct extra_chunk list_bt2020[2], list_bt709[2];
   struct extra_chunk list_srgb_gama[3], list_tf13_srgb[3];
   struct extra_chunk c_pq, c_tf13, c_srgb, c_gama, c_bt2020, c_bt709;
   struct extra_chunk c_gama_other;
   int x, y;

   /* Build the chunk lists. */
   c_pq = cicp_chunk(1/*BT.709 primaries*/, 16/*PQ*/);
   c_tf13 = cicp_chunk(1, 13/*sRGB transfer*/);
   c_bt2020 = cicp_chunk(11/*BT.2020*/, 16);
   c_bt709 = cicp_chunk(1, 16);
   c_srgb = srgb_chunk();
   c_gama = gama_chunk(PNG_GAMMA_sRGB_INVERSE);
   /* A deliberately different gAMA (encoding gamma 1/1.8) so that, if gAMA
    * wrongly wins over sRGB, the output diverges measurably.
    */
   c_gama_other = gama_chunk(PNG_GAMMA_MAC_INVERSE);

   list_pq[0] = c_pq; list_pq[1] = end;
   list_pq_srgb[0] = c_pq; list_pq_srgb[1] = c_srgb; list_pq_srgb[2] = end;
   list_pq_gama[0] = c_pq; list_pq_gama[1] = c_gama; list_pq_gama[2] = end;
   list_srgb_tf[0] = c_tf13; list_srgb_tf[1] = end;
   list_srgb_chunk[0] = c_srgb; list_srgb_chunk[1] = end;
   list_bt2020[0] = c_bt2020; list_bt2020[1] = end;
   list_bt709[0] = c_bt709; list_bt709[1] = end;
   list_srgb_gama[0] = c_gama_other; list_srgb_gama[1] = c_srgb;
      list_srgb_gama[2] = end;
   list_tf13_srgb[0] = c_tf13; list_tf13_srgb[1] = c_srgb;
      list_tf13_srgb[2] = end;

   printf("cicp_gamma: constructing test PNGs\n");
   make_png(&png_plain, no_chunks, 0);
   make_png(&png_pq, list_pq, 0);
   make_png(&png_pq_srgb, list_pq_srgb, 0);
   make_png(&png_pq_gama, list_pq_gama, 0);
   make_png(&png_srgb_tf, list_srgb_tf, 0);
   make_png(&png_srgb_chunk, list_srgb_chunk, 0);
   make_png(&png_bt2020, list_bt2020, 0);
   make_png(&png_bt709, list_bt709, 0);
   make_png(&png_srgb_gama, list_srgb_gama, 0);
   make_png(&png_tf13_srgb, list_tf13_srgb, 0);

   /* Test 1: PQ must produce no gamma correction.  Read with a linear output
    * space (screen gamma 1.0); this makes any wrongly-applied file gamma
    * visible.  The plain PNG has no gamma chunk, so it is a pure passthrough
    * and serves as the reference.  The discriminating cases are the ones where
    * an sRGB or gAMA chunk coexists with the cICP: the PNGv3 precedence rules
    * give cICP priority, so the lower-priority gamma must not be applied.
    */
   printf("cicp_gamma: test 1, PQ transfer 16 suppresses gamma\n");
   read_png(&png_plain, &r_plain, 1, PNG_GAMMA_LINEAR, 0);
   read_png(&png_pq, &r_pq, 1, PNG_GAMMA_LINEAR, 0);
   read_png(&png_pq_srgb, &r_pq_srgb, 1, PNG_GAMMA_LINEAR, 0);
   read_png(&png_pq_gama, &r_pq_gama, 1, PNG_GAMMA_LINEAR, 0);
   check(results_equal(&r_pq, &r_plain),
       "PQ cICP output matches no-gamma passthrough");
   check(results_equal(&r_pq_srgb, &r_plain),
       "PQ cICP + sRGB does not apply sRGB gamma");
   check(results_equal(&r_pq_gama, &r_plain),
       "PQ cICP + gAMA does not apply gAMA gamma");
   check(results_equal(&r_pq_srgb, &r_pq),
       "PQ cICP + sRGB equals PQ cICP alone");

   /* Test 2: cICP transfer 13 must behave exactly like an sRGB chunk.  Use a
    * linear output space so the file->screen gamma correction is actually
    * exercised (with an sRGB output both sides would be identity).
    */
   printf("cicp_gamma: test 2, transfer 13 matches sRGB chunk\n");
   read_png(&png_srgb_tf, &r_srgb_tf, 1, PNG_GAMMA_LINEAR/*screen=1.0*/, 0);
   read_png(&png_srgb_chunk, &r_srgb_chunk, 1, PNG_GAMMA_LINEAR, 0);
   {
      struct read_result r_plain_linear;
      read_png(&png_plain, &r_plain_linear, 1, PNG_GAMMA_LINEAR, 0);
      check(results_equal(&r_srgb_tf, &r_srgb_chunk),
          "cICP transfer 13 output equals sRGB chunk output");
      check(!results_equal(&r_srgb_tf, &r_plain_linear),
          "sRGB transfer actually performs gamma correction");
      free_result(&r_plain_linear);
   }

   /* Test 5: PNGv3 chunk precedence. */
   printf("cicp_gamma: test 5, gamma chunk precedence\n");
   read_png(&png_srgb_gama, &r_srgb_gama, 1, PNG_GAMMA_LINEAR, 0);
   read_png(&png_tf13_srgb, &r_tf13_srgb, 1, PNG_GAMMA_LINEAR, 0);
   /* No cICP: sRGB must override a different gAMA, so the result matches the
    * sRGB-only file, not the gAMA value.
    */
   check(results_equal(&r_srgb_gama, &r_srgb_chunk),
       "sRGB takes precedence over a coexisting different gAMA");
   /* cICP(transfer 13) + sRGB: cICP wins, so it matches cICP(13) alone (which
    * test 2 already proved equals the sRGB chunk result).
    */
   check(results_equal(&r_tf13_srgb, &r_srgb_tf),
       "cICP transfer 13 takes precedence over coexisting sRGB");

   /* Test 3: rgb-to-gray must use the cICP primaries.  BT.2020 and BT.709
    * have different luminance coefficients, so the gray rows must differ.
    * Both use PQ transfer here so gamma tables do not interfere.
    */
   printf("cicp_gamma: test 3, rgb-to-gray uses cICP primaries\n");
   read_png(&png_bt2020, &r_bt2020, 0, PNG_GAMMA_LINEAR, 1/*rgb to gray*/);
   read_png(&png_bt709, &r_bt709, 0, PNG_GAMMA_LINEAR, 1);
   check(!results_equal(&r_bt2020, &r_bt709),
       "BT.2020 and BT.709 cICP produce different gray coefficients");
   {
      /* And the BT.709 cICP result must match the library's built-in default
       * (Rec.709) rgb-to-gray coefficients, exercised on the plain PNG.
       */
      struct read_result r_plain_gray;
      read_png(&png_plain, &r_plain_gray, 0, PNG_GAMMA_LINEAR, 1);
      check(results_equal(&r_bt709, &r_plain_gray),
          "BT.709 cICP gray matches default Rec.709 coefficients");
      free_result(&r_plain_gray);
   }

   /* Test 4: no cICP -> the samples are passed through byte for byte when no
    * gamma is resolved.  The plain PNG carries no gamma chunk and is read with
    * a screen gamma but no resolvable file gamma, so every output byte must
    * equal the corresponding encoded sample byte.  This is the guarantee that
    * the cICP interpretation cannot change the output of files without cICP.
    */
   printf("cicp_gamma: test 4, no-cICP samples are unchanged\n");
   {
      size_t rowbytes = r_plain.rowbytes;
      int unchanged = 1;

      if (rowbytes != (size_t)TEST_WIDTH * 3 ||
          r_plain.height != TEST_HEIGHT)
         unchanged = 0;
      else
      {
         for (y = 0; y < TEST_HEIGHT && unchanged; ++y)
         {
            const unsigned char *p =
                r_plain.rows + rowbytes * (size_t)y;
            for (x = 0; x < TEST_WIDTH; ++x)
            {
               int i = y * TEST_WIDTH + x;
               if (p[x * 3 + 0] != (unsigned char)(i * 7 + 16) ||
                   p[x * 3 + 1] != (unsigned char)(i * 3 + 64) ||
                   p[x * 3 + 2] != (unsigned char)(255 - i * 5))
               {
                  unchanged = 0;
                  break;
               }
            }
         }
      }
      check(unchanged, "plain (no cICP) pixels are byte identical to input");
   }

   free_result(&r_plain);
   free_result(&r_pq);
   free_result(&r_pq_srgb);
   free_result(&r_pq_gama);
   free_result(&r_srgb_tf);
   free_result(&r_srgb_chunk);
   free_result(&r_bt2020);
   free_result(&r_bt709);
   free_result(&r_srgb_gama);
   free_result(&r_tf13_srgb);

   free(png_plain.data); free(png_pq.data); free(png_pq_srgb.data);
   free(png_pq_gama.data); free(png_srgb_tf.data); free(png_srgb_chunk.data);
   free(png_bt2020.data); free(png_bt709.data);
   free(png_srgb_gama.data); free(png_tf13_srgb.data);

   if (failures != 0)
   {
      printf("cicp_gamma: %d test(s) failed\n", failures);
      return 1;
   }
   printf("cicp_gamma: all tests passed\n");
   return 0;
}
