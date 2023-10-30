## References:
# https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
# https://developers.google.com/earth-engine/tutorials/community/detecting-changes-in-sentinel-1-imagery-pt-1

#%% Import necessary libraries
import ee
import time
from datetime import date, timedelta

#%% Function to get S-2 images with cloud cover estimate

def get_s2_sr_cld_col(aoi, start_date, end_date, cloud_filter):
    """Function to get Sentinel-2 images and stack it with probability of cloud
    cover"""
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_filter)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

#%% Function to add cloud probability and mask to S-2 images

def add_cloud_bands(img):
    """Add s2cloudless probability layer and cloud mask as layers to the S-2
    image"""
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))

#%% Function to add dark pixels for cloud shadows

def add_shadow_bands(img):
    """ Add dark pixels, cloud projection, and identified shadows as bands"""
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

#%% Add cloud shadow mask to the image set

def add_cld_shdw_mask(img):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img.addBands(is_cld_shdw)

#%% Apply cloud shadow mask to the images

def apply_cld_shdw_mask(img):
    """ Apply cloud mask to each band"""
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)

#%% Function to get string of shifted date

def shiftedDate(referenceDate, TmDelta):
    t = time.strptime(referenceDate, '%Y-%m-%d')
    modDate = date(t.tm_year, t.tm_mon, t.tm_mday) + timedelta(TmDelta)
    return modDate.strftime('%Y-%m-%d')

#%% Main

if __name__ == '__main__':
    
    ee.Initialize()
    
    locationDict = {'BethaniaAUS': [153.155, -27.684, '2022-03-01', '2022-03-11'], 
                    'TrentonUS': [-74.745, 40.207, '2021-09-02', '2021-09-12'], 
                    'AmosUS': [-89.807, 30.352, '2021-08-30', '2021-09-09'],
                    'SilcharIND': [92.779, 24.833, '2022-06-20', '2022-06-30'],
                    'SurOMAN': [59.508, 22.568, '2021-07-17', '2021-07-27'],
                    'LadysmithSA': [29.779, -28.545, '2022-01-17', '2022-01-27']}
    
    
    for key, val in locationDict.items():
        
        ## Your city's location and filters
        location = key
        AOI = ee.Geometry.Point(val[0], val[1])
        START_DATE = val[2]
        END_DATE = val[3]
        CLOUD_FILTER = 100
        CLD_PRB_THRESH = 40
        NIR_DRK_THRESH = 0.15
        CLD_PRJ_DIST = 4
        BUFFER = 300
        
        ## BF date and images
        BFF_START_DATE = shiftedDate(START_DATE, -365)
        BFF_END_DATE = shiftedDate(START_DATE, -60)
        BFF_CLOUD_FILTER = 0.5
        
        ## Create 23 km buffer around AOI point
        bufferedAOI_for_filtering = AOI.buffer(23000)
        ## Create 16 km buffer around AOI point for downloading
        bufferedAOI = AOI.buffer(16000)
        # projection = s2_img.projection().getInfo()
        
        s2_sr_cld_col_bff = get_s2_sr_cld_col(bufferedAOI_for_filtering, BFF_START_DATE, BFF_END_DATE, BFF_CLOUD_FILTER)
        # Sort the collection by ascending order of sensing time, since the subsequent mosaic() operation puts the last layer on top
        s2_sr_cld_col_bff_sorted = s2_sr_cld_col_bff.sort('system:time_start', True)
        S2_BFF_img = s2_sr_cld_col_bff_sorted.mosaic()
        # Get the last image from the collection
        s2_bff_lastImg = s2_sr_cld_col_bff.limit(1, 'system:time_start', False).first()
        s2_bff_lastImgDate = ee.Date.format(ee.Date(s2_bff_lastImg.get('system:time_start'))).getInfo()
        print('Location: {}'.format(location))
        print('First S2 AFF image(s) was obtained between ' + START_DATE + ' to ' + END_DATE)
        print('Last S2 BFF cloud-free image was obtained at ' + s2_bff_lastImgDate)
        s2_bff_lastImgDate = s2_bff_lastImgDate.replace(':', '_')
        
        ## If happy with evaluation, use same image
        s2_sr_cld_col = get_s2_sr_cld_col(bufferedAOI_for_filtering, START_DATE, END_DATE, CLOUD_FILTER)
        s2_img = s2_sr_cld_col.map(add_cld_shdw_mask).median()
        not_cloudmask = s2_img.select('cloudmask').Not()
        s2_sr_min = s2_sr_cld_col.map(add_cld_shdw_mask).map(apply_cld_shdw_mask).min() # Using minimum of band values here since water emits least radiation
        
        ## Export image to drive
        taskConfig1 = {'image':s2_img.select('B.*'), 'region':bufferedAOI, 'fileNamePrefix':'AFFRawImgAllBands', 'folder':'GEE_' + location + '_S2_AFF' + START_DATE + 'to' + END_DATE, 'scale':20, 'description':(location + 'S2AFFRawImgAllBands')}
        taskConfig2 = {'image':s2_img.select('TCI_.*'), 'region':bufferedAOI, 'fileNamePrefix':'AFFRawImgTCI', 'folder':'GEE_' + location + '_S2_AFF' + START_DATE + 'to' + END_DATE, 'scale':20, 'description':(location + 'S2AFFRawImgTCI')}
        taskConfig3 = {'image':not_cloudmask, 'region':bufferedAOI, 'fileNamePrefix':'AFF_notcloudmask', 'folder':'GEE_' + location + '_S2_AFF' + START_DATE + 'to' + END_DATE, 'scale':20, 'description':(location + 'S2not_cloudmask')}
        taskConfig4 = {'image':s2_sr_min, 'region':bufferedAOI, 'fileNamePrefix':'AFF_cloudFreeBands', 'folder':'GEE_' + location + '_S2_AFF' + START_DATE + 'to' + END_DATE, 'scale':20, 'description':(location + 'S2AFF_cloudFreeBands')}
        taskConfig5 = {'image':S2_BFF_img.select('TCI_.*'), 'region':bufferedAOI, 'fileNamePrefix':'BFFRawImgTCI', 'folder':'GEE_' + location + '_S2_BFF' + s2_bff_lastImgDate, 'scale':20, 'description':(location + 'S2BFF_TCI')}
        taskConfig6 = {'image':S2_BFF_img.select('B.*'), 'region':bufferedAOI, 'fileNamePrefix':'BFF_cloudFreeBands', 'folder':'GEE_' + location + '_S2_BFF' + s2_bff_lastImgDate, 'scale':20, 'description':(location + '_S2BFF_cloudFreeBands')}
        
        
        ## S-1 Images
        
        S1_AFF_img_col = (ee.ImageCollection('COPERNICUS/S1_GRD')
                      .filterBounds(bufferedAOI_for_filtering)
                      .filterMetadata('instrumentMode', 'equals', 'IW')
                      .filterDate(ee.Date(START_DATE), ee.Date(shiftedDate(START_DATE, 12))))
        # Sort so that the first image is last in the collection, since the subsequent mosaic() operation puts the last layer on top
        S1_AFF_img_col_sorted = S1_AFF_img_col.sort('system:time_start', False)
        S1_AFF_img = S1_AFF_img_col_sorted.mosaic()
        
        S1_AFF_img_startDate = ee.Date.format(ee.Date(S1_AFF_img_col.first().get('system:time_start'))).getInfo()
        S1_AFF_img_endDate = ee.Date.format(ee.Date(S1_AFF_img_col_sorted.first().get('system:time_start'))).getInfo()
        print('AFF S-1 images were obtained between ' + S1_AFF_img_startDate + ' and ' + S1_AFF_img_endDate)
        S1_AFF_imgDate = (S1_AFF_img_startDate + 'to' + S1_AFF_img_endDate).replace(':', '_')
        
        # Get orbit properties from the AFF images so we can narrow down the search for BFF images with only those properties for repeatability
        orbitPassDir = [dictItem['properties']['orbitProperties_pass'] for dictItem in S1_AFF_img_col_sorted.getInfo()['features']]
        relOrbitNum = [dictItem['properties']['relativeOrbitNumber_start'] for dictItem in S1_AFF_img_col_sorted.getInfo()['features']]
        # Get S-1 BFF images close to when S-2 BFF images are collected
        S1_BFF_img_col = (ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(bufferedAOI_for_filtering)
        .filterDate(ee.Date(shiftedDate(s2_bff_lastImgDate.split('T')[0], -12)), ee.Date(s2_bff_lastImgDate.split('T')[0]))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.inList('orbitProperties_pass', ee.List(orbitPassDir)))
        .filter(ee.Filter.inList('relativeOrbitNumber_start', ee.List(relOrbitNum))))
        # Sort the collection by ascending order of sensing time, since the subsequent mosaic() operation puts the last layer on top
        S1_BFF_img_col_sorted = S1_BFF_img_col.sort('system:time_start', True)
        S1_BFF_img = S1_BFF_img_col_sorted.mosaic()
                               
        S1_BFF_img_startDate = ee.Date.format(ee.Date(S1_BFF_img_col.first().get('system:time_start'))).getInfo()
        S1_BFF_img_endDate = ee.Date.format(ee.Date(S1_BFF_img_col.limit(1, 'system:time_start', False).first().get('system:time_start'))).getInfo()
        print('BFF S-1 images were obtained between ' + S1_BFF_img_startDate + ' and ' + S1_BFF_img_endDate)
        S1_BFF_imgDate = (S1_BFF_img_startDate + 'to' + S1_BFF_img_endDate).replace(':', '_')
        
        taskConfig7 = {'image':S1_AFF_img.select('V.*'), 'region':bufferedAOI, 'fileNamePrefix':'AFF_dB', 'folder':'GEE_' + location + '_S1_AFF' + S1_AFF_imgDate, 'scale':10, 'description':(location + '_S1_AFF')}
        taskConfig8 = {'image':S1_BFF_img.select('V.*'), 'region':bufferedAOI, 'fileNamePrefix':'BFF_dB', 'folder':'GEE_' + location + '_S1_BFF' + S1_BFF_imgDate, 'scale':10, 'description':(location + '_S1_BFF')}
        
        ## Write data to G Drive
        
        taskConfigList = [taskConfig1, taskConfig2, taskConfig3, taskConfig4, taskConfig5, taskConfig6, taskConfig7, taskConfig8]
        
        for ii in range(len(taskConfigList)):
            exportTask = ee.batch.Export.image.toDrive(**taskConfigList[ii])
            exportTask.start()
            exportTask.status()
            while exportTask.active():
                print('Waiting on (description: {}).'.format(exportTask.status()['description']))
                time.sleep(10)
            print(exportTask.status()['state'])

