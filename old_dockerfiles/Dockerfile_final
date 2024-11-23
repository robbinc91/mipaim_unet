FROM tensorflow/tensorflow:2.0.0-py3

ARG keras=2.3.1
ARG pydicom=2.0.0
ARG h5py=2.9.0
ARG nibabel=3.1.1

LABEL robex=1.2 ants=2.3.1 mni=icbm152_2009c_1mm fsl=6.0.0 keras=${keras} \
    pydicom=${pydicom} h5py=${h5py} nibabel=${nibabel}
	

# Tools	
ADD preprocessing_utils/ants 								/opt/ants
ADD preprocessing_utils/ROBEX 								/opt/ROBEX
ADD preprocessing_utils/mni 								/opt/mni
ADD preprocessing_utils/n4 									/opt/n4
ADD preprocessing_utils/fsl 								/opt/fsl
ADD preprocessing_utils/pipeline 							/opt/pipeline

# our codes
ADD process.sh       					/usr/local/bin/process.sh
ADD utils                   			/usr/local/bin/utils
ADD predict.py              			/usr/local/bin/predict.py
ADD ready_models/cerebellum_full.h5  	/opt/models/cerebellum_full.h5

RUN apt-get update && apt-get install -y unzip libopenblas-dev

RUN pip3 install --no-cache-dir deepbrain keras==${keras} pydicom==${pydicom} h5py==${h5py} nibabel==${nibabel} opencv-python

ENV PATH=/opt/ants/bin:/opt/fsl/bin:/opt/ROBEX:/opt/mni/scripts:/opt/n4/scripts:/opt/pipeline:$PATH

ENTRYPOINT ["/usr/local/bin/process.sh"]
CMD ["-h"]
