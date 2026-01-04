#!/bin/bash

NUMBER_OF_ELEMENTS=""
DEVICE=""
TEST_TYPE=""
EXEC_NAME="stream_test"

while [ "$1" != "" ]; do
  case $1 in
    --n )
        NUMBER_OF_ELEMENTS="-n $2"
        shift
        ;;
    --d )
        DEVICE="-d $2"
        shift
        ;;
    --t )
        TEST_TYPE="-t $2"
        shift
        ;;
    --dt )
        DATA_TYPE="$2"
        if [ "$2" == "fp32" ]; then
            EXEC_NAME="stream_test_fp32"
        fi
        shift
        ;;
  esac
  shift
done

SCRIPT_DIR=$( cd -- "$( dirname -- "$( readlink -f "${BASH_SOURCE[0]}" )" )" &> /dev/null && pwd )
STREAM_EXEC="$SCRIPT_DIR/$EXEC_NAME"

echo "Command line: $STREAM_EXEC $DEVICE $NUMBER_OF_ELEMENTS $TEST_TYPE"
$STREAM_EXEC $DEVICE $NUMBER_OF_ELEMENTS $TEST_TYPE
