#!/bin/bash -e

mkdir -p 3rd_party
cd 3rd_party

echo "==================== Installing libxc ===================="
if [ -f libxc-2.0.1.tar.gz ]; then
  echo "Installation already started, skipping it."
else
  wget http://www.cp2k.org/static/downloads/libxc-2.0.1.tar.gz
  echo "c332f08648ec2bc7ccce83e45a84776215aa5dfebc64fae2a23f2ac546d41ea4 *libxc-2.0.1.tar.gz" | sha256sum  --check
  tar -xzf libxc-2.0.1.tar.gz
  cd libxc-2.0.1
  mkdir -p install
  ./configure  --prefix=${PWD}/install
  make -j 20
  make install
  cd ..
fi

echo "==================== Installing libint ===================="
if [ -f libint-1.1.4.tar.gz ]; then
  echo "Installation already started, skipping it."
else
  wget http://www.cp2k.org/static/downloads/libint-1.1.4.tar.gz
  echo "f67b13bdf1135ecc93b4cff961c1ff33614d9f8409726ddc8451803776885cff *libint-1.1.4.tar.gz" | sha256sum  --check
  tar -xzf libint-1.1.4.tar.gz
  cd libint-1.1.4
  mkdir -p install
  ./configure  --prefix=${PWD}/install --with-libint-max-am=5 --with-libderiv-max-am1=4
  make -j 20
  make install
  cd ..
fi

echo "==================== Installing FFTW ===================="
if [ -f fftw-3.3.4.tar.gz ]; then
  echo "Installation already started, skipping it."
else
  wget http://www.cp2k.org/static/downloads/fftw-3.3.4.tar.gz
  echo "8f0cde90929bc05587c3368d2f15cd0530a60b8a9912a8e2979a72dbe5af0982 *fftw-3.3.4.tar.gz" | sha256sum  --check 
  tar -xzf fftw-3.3.4.tar.gz
  cd fftw-3.3.4
  mkdir -p install
  ./configure  --prefix=${PWD}/install --enable-openmp
  make -j 20
  make install
  cd ..
fi

echo "==================== Installing ELPA ===================="
if [ -f elpa-2013.11.008.tar.gz ]; then
  echo "Installation already started, skipping it."
else
  wget http://www.cp2k.org/static/downloads/elpa-2013.11.008.tar.gz
  echo "d4a028fddb64a7c1454f08b930525cce0207893c6c770cb7bf92ab6f5d44bd78 *elpa-2013.11.008.tar.gz" | sha256sum  --check
  tar -xzf elpa-2013.11.008.tar.gz
  cd ELPA_2013.11
  mkdir -p install
  ./configure  --prefix=${PWD}/install --enable-openmp --with-generic
  make -j 20
  make install
  cd ..
fi

#EOF
