#ifndef PACKED_STATUS_H
#define PACKED_STATUS_H

#include <cassert>
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define FARM_STATUS_HD __host__ __device__
#define FARM_STATUS_INLINE __forceinline__
#else
#define FARM_STATUS_HD
#define FARM_STATUS_INLINE inline
#endif

FARM_STATUS_HD FARM_STATUS_INLINE uint16_t status_category_count(uint8_t bitwidth)
{
	return static_cast<uint16_t>(1U << bitwidth);
}

FARM_STATUS_HD FARM_STATUS_INLINE uint8_t status_max_value(uint8_t bitwidth)
{
	return static_cast<uint8_t>(status_category_count(bitwidth) - 1U);
}

FARM_STATUS_HD FARM_STATUS_INLINE size_t packed_status_bytes(size_t pixel_count, uint8_t bitwidth)
{
	return (pixel_count * bitwidth + 7U) / 8U;
}

FARM_STATUS_HD FARM_STATUS_INLINE uint8_t read_packed_status(
	const uint8_t *data, uint32_t pixel_id, uint8_t bitwidth)
{
	if (bitwidth == 8)
	{
		return data[pixel_id];
	}
	if (bitwidth == 4)
	{
		const uint8_t value = data[pixel_id >> 1U];
		return static_cast<uint8_t>((value >> ((pixel_id & 1U) * 4U)) & 0x0fU);
	}
	if (bitwidth == 2)
	{
		const uint8_t value = data[pixel_id >> 2U];
		return static_cast<uint8_t>((value >> ((pixel_id & 3U) * 2U)) & 0x03U);
	}

	const uint32_t bit_offset = pixel_id * static_cast<uint32_t>(bitwidth);
	const uint32_t byte_index = bit_offset >> 3U;
	const uint32_t shift = bit_offset & 7U;
	uint16_t value = data[byte_index];
	if (shift + bitwidth > 8U)
	{
		value |= static_cast<uint16_t>(data[byte_index + 1U]) << 8U;
	}
	const uint16_t mask = static_cast<uint16_t>((1U << bitwidth) - 1U);
	return static_cast<uint8_t>((value >> shift) & mask);
}

inline void write_packed_status(
	uint8_t *data, uint32_t pixel_id, uint8_t bitwidth, uint8_t status)
{
	assert(bitwidth >= 2 && bitwidth <= 8);
	assert(status <= status_max_value(bitwidth));

	if (bitwidth == 8)
	{
		data[pixel_id] = status;
		return;
	}
	if (bitwidth == 4)
	{
		const uint32_t byte_index = pixel_id >> 1U;
		const uint32_t shift = (pixel_id & 1U) * 4U;
		data[byte_index] = static_cast<uint8_t>(
			(data[byte_index] & ~(0x0fU << shift)) | (status << shift));
		return;
	}
	if (bitwidth == 2)
	{
		const uint32_t byte_index = pixel_id >> 2U;
		const uint32_t shift = (pixel_id & 3U) * 2U;
		data[byte_index] = static_cast<uint8_t>(
			(data[byte_index] & ~(0x03U << shift)) | (status << shift));
		return;
	}

	const uint32_t bit_offset = pixel_id * static_cast<uint32_t>(bitwidth);
	const uint32_t byte_index = bit_offset >> 3U;
	const uint32_t shift = bit_offset & 7U;
	const uint16_t value_mask = static_cast<uint16_t>((1U << bitwidth) - 1U);
	const uint16_t shifted_mask = static_cast<uint16_t>(value_mask << shift);
	uint16_t value = data[byte_index];
	if (shift + bitwidth > 8U)
	{
		value |= static_cast<uint16_t>(data[byte_index + 1U]) << 8U;
	}
	value = static_cast<uint16_t>(
		(value & ~shifted_mask) | ((status & value_mask) << shift));
	data[byte_index] = static_cast<uint8_t>(value & 0xffU);
	if (shift + bitwidth > 8U)
	{
		data[byte_index + 1U] = static_cast<uint8_t>(value >> 8U);
	}
}

#undef FARM_STATUS_HD
#undef FARM_STATUS_INLINE

#endif
