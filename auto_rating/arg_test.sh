#!/usr/bin/env bash

function dir2name() {
    local  __result_var=$1
    dir_basename="$(basename -- "${2}")"
    split=(${dir_basename//_/ })
    eval ${__result_var}="${split[0]}_${split[1]}_epoch_${split[6]}"
}

function dir2test() {
    local  __result_var=$1
    dir_basename="$(basename -- "${2}")"
    split=(${dir_basename//_/ })
    eval ${__result_var}="${split[0]}"
}


echo $#
args=("$@")
dir2name outDir "${args[0]}"
for dir in "${args[@]:1}"; do
    echo "${dir}"
    dir2name tmp "${dir}"
    outDir+=" vs ${tmp}"
done

dir2test test "${args[0]}"
arr=("${test}")
for dir in "${args[@]:1}"; do
    dir2test tmp "${dir}"
    arr+=("${tmp}")
done

echo $(sed "s/ /--/g" <<< "${arr[@]}")

echo $outDir
dir2name tmp $1
echo $tmp

