import numpy as np
import struct

def convert_ply_to_points3D_bin(self, output_file, ply_data):
    num_valid_points3D = sum(
        1 for point3D_idx in self.point3D_id_to_point3D_idx.itervalues()
        if point3D_idx != np.uint64(-1))

    iter_point3D_id_to_point3D_idx = \
        self.point3D_id_to_point3D_idx.iteritems()

    with open(output_file, 'wb') as fid:
        fid.write(struct.pack('L', num_valid_points3D))

        for point3D_id, point3D_idx in iter_point3D_id_to_point3D_idx:
            if point3D_idx == np.uint64(-1):
                continue

            fid.write(struct.pack('L', point3D_id))
            fid.write(self.points3D[point3D_idx].tobytes())
            fid.write(self.point3D_colors[point3D_idx].tobytes())
            fid.write(self.point3D_errors[point3D_idx].tobytes())
            fid.write(
                struct.pack('L', len(self.point3D_id_to_images[point3D_id])))
            fid.write(self.point3D_id_to_images[point3D_id].tobytes())