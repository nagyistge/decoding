"""
do encoding model across sessions

"""
from pyneurovault import api
import os,glob,sys,ctypes
import sklearn.linear_model
from glob import glob
import nibabel
import numpy
import os

download_folder = '/home/vanessa/Documents/Work/COGAT/decoding'
standard = "/usr/share/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz"
tstat_directory = "/home/vanessa/Documents/Work/COGAT/decoding/tstat"

def get_coding():
    return pandas.read_csv("data_matrix.tsv",sep="\t",index_col=0)


# Download NeuroVault image files
def download_files(coding,download_folder,standard):
    if not os.path.exists("%s/resampled" %(download_folder)):
        nv = api.NeuroVault()
        nv.download_images(dest_dir=download_folder,
                       target=standard,
                       image_ids=coding.image_id.tolist())
    files = glob("%s/resampled/*.nii.gz" %(download_folder))

    # Only return images that we are able to download (images get deleted in NV)
    image_ids = [int(f.split("/")[-1].replace(".nii.gz","")) for f in files]
    missings = numpy.setdiff1d(coding.image_id.tolist(),image_ids)
    for missing in missings:
        coding = coding[coding.image_id != missing]
    return coding,files


def make_binary_labels(coding):
    labels = coding.index
    binary_labels = []
    lookup = dict()
    taskcount = 0
    lasttask = ""
    for label in labels:
        task,contrast = [x.strip(" ") for x in label.split("::")]
        if lasttask != task:
            taskcount = taskcount + 1
        if taskcount not in lookup:
            contrastcount = 1
            lookup[taskcount] = task
        else:
            contrastcount = contrastcount + 1
        lasttask = task
        binary_labels.append([taskcount,contrastcount])
    return binary_labels
            
        
                

def load_data(coding,files,standard):
    # Generate a mask from the standard
    mask = nibabel.load(standard).get_data()
    mask[mask!=0] = 1

    # Data matrix
    length = len(mask[mask==1])
    contrastdata = numpy.zeros((len(files),length))

    for f in range(len(files)):
        contrastdata[f,:] = nibabel.load(files[f]).get_data()[mask==1]
    contrastdata = pandas.DataFrame(contrastdata)
    
    # Be sure we have right row names based on image id
    imageids = [os.path.basename(f).strip(".nii.gz").replace("0","") for f in files]
    contrastdata.index = imageids

    return contrastdata                      

def get_design_matrix(coding,binary_labels):
    desmtx = coding.copy()
    desmtx.index = ["%s-%s" %(c[0],c[1]) for c in binary_labels]
    desmtx = desmtx.drop('files', 1)
    desmtx = desmtx.drop('image_id', 1)
    return desmtx


def save_nifti(tstat_df,output_prefix,standard):
    standard_brain = nibabel.load(standard)
    for row in tstat_df.iterrows(): 
       output_name = "%s_%s.nii.gz" %(output_prefix,row[0].replace(" ","_"))
       empty_nii = numpy.zeros(standard_brain.shape)
       empty_nii[standard_brain.get_data()!=0]=row[1]
       tmp = nibabel.Nifti1Image(empty_nii,affine=standard_brain.get_affine(),header=standard_brain.get_header())
       nibabel.save(tmp,output_name)

if __name__=="__main__":
 
    coding = get_coding()
    files =  download_files(coding,download_folder,standard)
    contrastdata = load_data(coding,files,standard)

    # Let's make numeric labels for tasks and contrasts
    binary_labels = make_binary_labels(coding)
    desmtx = get_design_matrix(coding,binary_labels)

    # Demean and get degrees of freedom
    desmtx=desmtx-numpy.mean(desmtx,0)
    df = desmtx.shape[0] - desmtx.shape[1]

    # Each row is a concept, and each column is a voxel
    # Each value in these matrices will be a t statistic for how well the concept
    # can predict the voxel value
    tstat_lasso = numpy.zeros((desmtx.shape[1],contrastdata.shape[1]))
    tstat = numpy.zeros((desmtx.shape[1],contrastdata.shape[1]))

    # Linear regression (lm) and lasso (lr)
    lm = sklearn.linear_model.LinearRegression()
    lr = sklearn.linear_model.Lasso(alpha=0.01)
    ctr = 0
    ctr2 = 0
    # Iterating through voxels
    for i in range(contrastdata.shape[1]):
        # Show a count of every 100
        if ctr == 100:
            ctr = 0
            ctr2 += 1
            print ctr2
        else:
            ctr+=1
            ctr2+=1

        # This is a data value for a single voxel across all contrasts
        y = contrastdata.loc[:,i] - numpy.mean(contrastdata.loc[:,i])

        # The design matrix has the same number of rows (contrasts) and
        # each column is a concept term.
        lr.fit(desmtx,y)
        # I think this means we are seeing if presence of any concept (each column)
        # can predict the voxel value (y)

        # The "prediction" is the right side, and the actual is the left
        # So the error is the residuals
        resid = y.tolist() - desmtx.dot(lr.coef_)

        # Sum of squared error?
        sse = numpy.dot(resid,resid)/float(df)

        # And is a tstatistic the coefficients divided by sum of squared error?
        tstat_lasso[:,i] = lr.coef_/sse

        lm.fit(desmtx,y)
        resid = y.tolist() - desmtx.dot(lm.coef_)
        sse = numpy.dot(resid,resid)/float(df)
        tstat[:,i] = lm.coef_/sse

    
    tstat[numpy.isnan(tstat)]=0
    tstat_lasso[numpy.isnan(tstat_lasso)]=0
        
    tstat_lasso_df = pandas.DataFrame(tstat_lasso,index=desmtx.columns)
    tstat_df = pandas.DataFrame(tstat,index=desmtx.columns)
    tstat_lasso_df.to_pickle("tstat_lasso.pkl")
    tstat_df.to_pickle("tstat.pkl")
    numpy.save(os.path.join('encoding_tstat_lasso.npy'),tstat_lasso)
    numpy.save(os.path.join('encoding_tstat.npy'),tstat)

    # For each concept image, we need to save a tstat image!
    save_nifti(tstat_lasso_df,"%s/tstat_lasso" %tstat_directory,standard)
    save_nifti(tstat_df,"%s/tstat" %tstat_directory,standard)
