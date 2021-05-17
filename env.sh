if [[ -z "$LEGUP_PATH" ]]; then
    echo "ERROR: Must provide LEGUP_PATH in the environment!" 1>&2
fi
if [[ -z "$AUTOPHASE_PATH" ]]; then
    echo "ERROR: Must provide AUTOPHASE_PATH in the environment!" 1>&2
fi

export LD_LIBRARY_PATH=$LEGUP_PATH/lib:$LD_LIBRARY_PATH
export PATH=$LEGUP_PATH/bin:$PATH

