#!/usr/bin/env bash

function get_status() {
    local  __result_var=$1
    file_basename_no_ext="${2%.*}"
    eval ${__result_var}="${file_basename_no_ext#*"${3}"_}"
}

function dir2name() {
    local  __result_var=$1
    dir_basename="$(basename -- "${2}")"
    split=(${dir_basename//_/ })
    eval ${__result_var}="${split[0]}_${split[1]}_epoch_${split[6]}"
}


if [ $(bc <<< "$#>=2") -eq 0 ]; then
    echo "At least two directories expected as input"
    exit 1
fi

dirs=("$@")

dir2name outDir "${dirs[0]}"
for dir in "${dirs[@]:1}"; do
    dir2name tmp "${dir}"
    outDir+=" vs ${tmp}"
done

if $([ -d "${outDir}" ]); then rm -r "${outDir}"; fi
mkdir "${outDir}"

shopt -s nullglob
for file_first in "${dirs[0]}"/*; do
    first_basename="$(basename -- "${file_first}")"
    file_id="$(echo "${first_basename}" | cut -d_ -f1-5)"
    get_status status_first "${first_basename}" "${file_id}"

    files=("${file_first}")
    statuses=("${status_first}")
    for dir in "${dirs[@]:1}"; do
        file_other="$(find ${dir} -name "${file_id}*")"
        if [ ! -z "${file_other}" ]; then
            other_basename=$(basename -- "${file_other}")
            get_status status_other "${other_basename}" "${file_id}"
            files+=("${file_other}")
            statuses+=("${status_other}")
        else
            continue 2
        fi
    done

    outfile="${outDir}/${file_id}_$(sed "s/ /--/g" <<< "${statuses[@]}").png"
    montage "${files[@]}" -tile "${#dirs[@]}"x1 -geometry +0+0 "${outfile}"
done
shopt -u nullglob