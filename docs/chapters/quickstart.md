# Quickstart

After installing precon3d into your local python enviroment (either clone through git or use pip), follow the instructions below to begin processing data.

> Using Fiji (Fiji Is Just ImageJ):
> Some configuration files will have an entry for your Fiji app (when a Fiji subroutine is used). For Windows, use the path to fiji.exe and for Macs, use the path to the Fiji.app/fiji. 


The outline below will give you the general steps for preparing a 3D reconstuction of your serial sectioning data.  

1. Turn the `.czi` image data into stitched mosiacs. See `precon3d stitcher --help`
2. Pad these images and downscale them so they are easier to view and manage. See `precon3d post_stitch --help` and `precon3d downscaler --help`
3. Align the downscaled images. See `precon3d aligner --help`

> There are useful file management utilites available through `precon3d utility --help`