import struct
import functools
import numpy as np
def idx(path):

	HEX_TO_TYPE = {
		0x08: np.uint8,
		0x09: np.int8,
		0x0B: np.int16,
		0x0C: np.int32,
		0x0D: np.float32,
		0x0E: np.float64
	}
	with open(path, "rb") as file:
		(zeros, data_type, num_dimensions) = struct.unpack(">HBB",file.read(4))
		#print(zeros)
		#print(data_type)
		#print(num_dimensions)

		dims = struct.unpack(">" + "I"*num_dimensions, file.read(4*num_dimensions))
		#print(dims)

		data = np.fromfile(file, HEX_TO_TYPE[data_type], -1).reshape(dims)
		#print(data.shape)
		#print(data[0])
		#Only for images, not labels
		#print(data[0].reshape(28*28))
		return data


if __name__ == "__main__":
	np.set_printoptions(linewidth=175)
	try:
		#idx("../data/mnist/train_images")
		idx("../data/wmu/wmu_test_images")
	except IOError as e:
		print("Failed to open file:", e)