# Downscale

Downscaling is the process of resizing images to reduce their resolution. This is often necessary when we need to match the in-plane image resolution to the slice thicknesses of the data. For example, if the original image resolution is 0.5 μm/pixel and the slice thickness is 5 μm, a downscale operation can be performed to decrease the resolution by a factor of 10, resulting in a new resolution of 5 μm/pixel.

### Importance of Downscaling

The downscaled volume of data will have isotropic (evenly shaped) voxels with a resolution of 5 μm/voxel (here, the in-plane resolution is adjusted to match the slice thickness). This isotropic resolution is crucial for accurate three-dimensional reconstructions and analyses, as it ensures that the dimensions of the voxels are uniform in all directions.

It is important to note that during the downscaling process, we do not infill any data. As a result, the voxel resolution is inherently limited by the slice thickness. This means that while downscaling can help achieve a more uniform voxel shape, the quality and detail of the data may be constrained by the original slice thickness.

By carefully managing the downscaling process, the datasets can be optimized for analysis while maintaining the integrity of the information contained within the images.