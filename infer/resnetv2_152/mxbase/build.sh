#!/bin/bash

export ASCEND_HOME=/usr/local/Ascend
export ASCEND_VERSION=nnrt/latest
export ARCH_PATTERN=.
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib/modelpostprocessors:${LD_LIBRARY_PATH}

mkdir -p build
cd build || exit

function make_plugin() {
    if ! cmake ..;
    then
      echo "cmake failed."
      return 1
    fi

    if ! (make);
    then
      echo "make failed."
      return 1
    fi

    return 0
}

if make_plugin;
then
  echo "INFO: Build successfully."
else
  echo "ERROR: Build failed."
fi

cd - || exit

