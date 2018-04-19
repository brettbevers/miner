FROM pyspark-notebook:latest

ADD ./requirements.txt /tmp/requirements.txt
RUN conda install --yes --file /tmp/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

#ADD ./conf/eggs /tmp/eggs
#ADD ./scripts/install_eggs.sh /tmp/install_eggs.sh
#RUN chmod +x /tmp/install_eggs.sh && /tmp/install_eggs.sh /tmp/eggs
#RUN rm -rf /tmp
#
#ADD ./conf/jars /usr/local/extra-classes