import numpy as np
import open3d as o3d

from pathlib import Path

from dataclasses import dataclass

@dataclass
class PlyData:
    """Represents a simple wraper data class for ply file

    Attributes:
        _data_dir: Dictionary of ply meshes
    """
    
    _data_dir: Path
    
    def __post_init__(self):
        
        try:
            ply_files = self._data_dir.glob('*.ply')
            if not ply_files:
                raise Exception(f'Directory: {ply_files} does not conitan ply files.')
        except:
            raise Exception(f'Directory: {ply_files} does not conitan ply files.')

        self.ply_files = list(ply_files)
        assert len(self.ply_files) == 1, "Directory contain more than one init point cloud."
        print(f"Init point cloud: {self.ply_files[0]}")
        self.pcd = o3d.io.read_point_cloud(str(self.ply_files[0]))

        
    @property
    def points(self):
        return np.asarray(self.pcd.points)
    
    @property
    def color(self):
        return np.asarray(self.pcd.colors)
    
    @property
    def point_ids(self):
        return np.arange(len(self.points()), dtype=np.uint64)
    
    @property
    def point3D_id_to_point3D_idx(self):
        p = dict(zip(self.points, self.point_ids))
        # for i in range(len(self.points)):
        #     p[self.points[i]] = i
        return p
    
    @property
    def point3D_errors(self):
        return np.zeros((len(self.points), 1), dtype=np.uint64)