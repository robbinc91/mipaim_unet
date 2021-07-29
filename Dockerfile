FROM preprocess/minimal:v1

#RUN pip3 install opencv-python

RUN rm -rf /opt/pipeline/perform_preprocess.sh
RUN rm -rf /opt/mni/scripts/perform_mni.sh
RUN rm -rf /usr/local/bin/preprocess.sh

COPY preprocessing_utils/mni/scripts/perform_mni.sh          /opt/mni/scripts/perform_mni.sh
COPY process.sh              /usr/local/bin/process.sh
COPY preprocessing_utils/pipeline/perform_preprocess.sh   /opt/pipeline/perform_preprocess.sh
COPY utils                   /usr/local/bin/utils
COPY predict.py              /usr/local/bin/predict.py
COPY ready_models/cerebellum_full.h5  /opt/models/cerebellum_full.h5

ENTRYPOINT ["/usr/local/bin/process.sh"]
CMD ["-h"]
