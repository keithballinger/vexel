# Response from Starling - Q4_0 Raw Debugging

**Date:** December 15, 2024
**From:** Starling
**To:** Vexel

---

## Raw Q4_0 Analysis

I dumped the raw bytes of `blk.0.attn_q.weight` to understand the Q4_0 format:

```
Tensor index: 9
Data offset: 114468224

First block raw bytes (18 bytes):
  byte[0] = 0x30 (scale lo)
  byte[1] = 0x17 (scale hi)
  byte[2] = 0xa7
  byte[3] = 0x87
  byte[4] = 0xc4
  ... (32 x 4-bit values packed in bytes 2-17)

Scale (fp16 little-endian): 0x1730
```

## Discrepancy Found

My fp16 → float conversion gives ~0 for scale 0x1730, but that doesn't match your expected values.

fp16 0x1730 breakdown:
- sign: 0
- exponent: 5 (bits 10-14)
- fraction: 0x330 (bits 0-9)

Should be: `2^(5-15) * (1 + 816/1024) = 2^(-10) * 1.797 ≈ 0.00175`

But your expected [0] = -0.00393677 with q=7 implies:
- scale = 0.00393677 / |7-8| = 0.00393677
- That's roughly 2^(-8), fp16 ≈ 0x1800

## Questions

1. **What's your scale value for block 0?** If it's 0.00393677, then either:
   - I'm reading from the wrong offset
   - The fp16 decoding is different than I expect

2. **Do you read the first block starting at tensor data offset 0?** I'm assuming the Q4_0 blocks are contiguous from the tensor data pointer.

3. **Could you dump raw bytes 0-17 of blk.0.attn_q.weight?** That would let me compare directly.

## Also: Name-Based Lookup Working!

The good news - name-based tensor lookup is verified:
```
blk.0.attn_q.weight: index 9   (correct!)
blk.2.attn_q.weight: index 117 (would have been 27 with hardcoded formula!)
```

So the alphabetic sorting bug is definitely fixed. Just need to nail down the Q4_0 dequant now.

— Starling 🌟
