from osgeo import gdal, ogr, gdalconst, osr
import numpy as np
import geopandas
import os
from sys import platform


def coords2rc(transform, coords):
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]
    col = int((coords[0] - xOrigin) / pixelWidth)
    row = int((yOrigin - coords[1]) / pixelHeight)
    return row, col


def rc2coords(transform, rc):
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]
    coordX = xOrigin + pixelWidth * (rc[1] + 0.5)
    coordY = yOrigin + pixelHeight * (rc[0] + 0.5)
    return coordX, coordY


def read_shp_point(filename):
    """Read the Point shapefile attributes as a list of xy coordinates"""
    result = list()
    startPoints = geopandas.read_file(filename)
    for pt in startPoints.geometry:
        result.append((pt.x, pt.y))
    return result


def read_shp_line(filename):
    """read multi-line shapefile as point coords which defining the lines"""
    output = list()
    lineshp = geopandas.read_file(filename)
    for ln in lineshp.geometry:
        output.append(list(zip(ln.xy[0], ln.xy[1])))
    return output


def gdal_read_prj():
    if platform == 'linux': # on Spartan
        prj_path = '/home/yuerongz/punim0728/WHProj/gis/projection.prj'
    else:
        prj_path = 'C:/Users/mike-u/Desktop/YUERONG/TUFLOW_WilliamHovell/model/gis/projection.prj'
    with open(prj_path, 'r') as f:
        prj_txt = f.read()
    srs = osr.SpatialReference()
    srs.ImportFromESRI([prj_txt])
    return srs.ExportToWkt()


def gdal_shpprojection(shpfile):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    projfile = driver.Open(shpfile)
    return projfile.GetLayer(0).GetSpatialRef().ExportToWkt()


def gdal_asarray(rasterfile):
    ds = gdal.Open(rasterfile)
    band = ds.GetRasterBand(1)
    return band.ReadAsArray()


def gdal_transform(rasterfile):
    return gdal.Open(rasterfile).GetGeoTransform()


def gdal_writetiff(arr_data, outfile, ras_temp=None, target_transform=None, target_projection=None):
    driver = gdal.GetDriverByName('GTiff')
    outdata = driver.Create(outfile, arr_data.shape[1], arr_data.shape[0], 1, gdal.GDT_Float32)
    if ras_temp is not None:
        ds = gdal.Open(ras_temp)
        outdata.SetGeoTransform(ds.GetGeoTransform())  ##sets same geotransform as temp
        outdata.SetProjection(ds.GetProjection())  ##sets same projection as temp
    elif all([target_transform, target_projection]):
        outdata.SetProjection(target_projection)
        outdata.SetGeoTransform(target_transform)
    elif ~bool(target_projection):
        outdata.SetProjection(gdal_read_prj())
        outdata.SetGeoTransform(target_transform)
    else:
        raise ValueError('Please provide a raster template or target transform (/&projection)!')
    outband = outdata.GetRasterBand(1)
    outband.WriteArray(arr_data)
    outdata.FlushCache()  ##saves to disk!!
    # return print('gdal_writeriff successful.')
    return 0


def gdal_asarray_crop_to(raster_file, ref_trans, ref_shape):
    arr = gdal_asarray(raster_file)
    arr_trans = gdal_transform(raster_file)
    r0, c0 = coords2rc(arr_trans, (ref_trans[0]+ref_trans[1]/2, ref_trans[3]-ref_trans[1]/2))
    arr = arr[r0:r0+ref_shape[0], c0:c0+ref_shape[1]]
    return arr


def gdal_writeasc(filename, arr_data, rastemp):
    format = "MEM"
    driver = gdal.GetDriverByName(format)
    dst_ds = driver.Create(filename, arr_data.shape[1], arr_data.shape[0], 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(rastemp.GetGeoTransform())
    dst_ds.SetProjection(rastemp.GetProjection())
    dst_ds.GetRasterBand(1).WriteArray(arr_data)
    format = 'AAIGrid'
    driver = gdal.GetDriverByName(format)
    dst_ds_new = driver.CreateCopy(filename, dst_ds)
    # del dst_ds
    # del dst_ds_new
    return print('gdal_writeasc successful!')


def raster_aggregate(srcfile, outfile, aggregate_level):
    src = gdal.Open(srcfile, gdalconst.GA_ReadOnly)
    dst_gt = list(src.GetGeoTransform())
    dst_gt[1] = dst_gt[1] * aggregate_level
    dst_gt[5] = dst_gt[5] * aggregate_level
    dst_gt = tuple(dst_gt)
    width = int(src.RasterXSize / aggregate_level)
    height = int(src.RasterYSize / aggregate_level)
    dst = gdal.GetDriverByName('GTiff').Create(outfile, width, height, 1, gdalconst.GDT_Float32)
    dst.SetGeoTransform(dst_gt)
    dst.SetProjection(src.GetProjection())
    gdal.ReprojectImage(src, dst, src.GetProjection(), None, gdalconst.GRA_Min)
    return 0


def flt2tif(fltfile, target_file):
    # fltfile = 'C:/Users/mike-u/Desktop/YUERONG/TUFLOW_WilliamHovell/results/iwl3/grids/WHYZ_iwl_h_070.flt'
    # target_file = 'tuflow_files_prep/WH_iwl_yz.tif'
    ds = gdal.Open(fltfile)
    arr_data = ds.ReadAsArray()
    arr_data[arr_data == -999] = np.nan
    driver = gdal.GetDriverByName('GTiff')
    outdata = driver.Create(target_file, arr_data.shape[1], arr_data.shape[0], 1, gdal.GDT_Float32)
    outdata.SetGeoTransform(ds.GetGeoTransform())  ##sets same geotransform as temp
    outdata.SetProjection(gdal_read_prj())  ##sets same projection as temp
    outband = outdata.GetRasterBand(1)
    outband.WriteArray(arr_data)
    outdata.FlushCache()  ##saves to disk!!

    # target_file = 'tuflow_files_prep/WH_iwl_yz.asc'
    # format = "MEM"
    # driver = gdal.GetDriverByName(format)
    # dst_ds = driver.Create(target_file, arr_data.shape[1], arr_data.shape[0], 1, gdal.GDT_Float32)
    # dst_ds.SetGeoTransform(ds.GetGeoTransform())
    # dst_ds.SetProjection(gdal_read_prj())
    # dst_ds.GetRasterBand(1).WriteArray(arr_data)
    # format = 'AAIGrid'
    # driver = gdal.GetDriverByName(format)
    # dst_ds_new = driver.CreateCopy(target_file, dst_ds)
    # del dst_ds
    # del dst_ds_new
    return 0

