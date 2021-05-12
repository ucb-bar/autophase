# LLVM Patches

## 1. Install LegUp LLVM and export the llvm path

```
export PATH=$LEGUP_PATH/llvm/Release+Asserts/bin:$PATH
```

## 2. Copy the patches to llvm for feature extraction
```
cp Analysis.cpp `llvm-config --prefix`/lib/Analysis/Analysis.cpp
cp Passes.h `llvm-config --prefix`/include/llvm/Analysis/Passes.h
cp GlobalVarPass.cpp `llvm-config --prefix`/lib/Analysis/GlobalVarPass.cpp
```      

## 3. Recompile LegUp LLVM
```
cd $LEGUP_PATH && make all
```
