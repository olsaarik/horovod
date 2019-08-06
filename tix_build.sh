#!/bin/bash
cd ./test && pip uninstall -y horovod && cd ..
rm -r ./build
python setup.py clean
python setup.py install
