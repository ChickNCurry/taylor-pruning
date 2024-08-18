FROM dustynv/l4t-ml:r32.7.1
WORKDIR /app
RUN pip3 install flask
RUN pip3 install torchinfo
RUN pip3 install matplotlib
RUN git clone https://github.com/NVIDIA-AI-IOT/jetcam && cd jetcam && python3 setup.py install
CMD ["python3", "-u", "/app/cam_server.py"]
EXPOSE 5000
