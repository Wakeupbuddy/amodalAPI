# evaluate the amodal segments
cd ./PythonAPI/
MAXPROP=1000
MODELNAME='example'
LOGNAME='log.txt'
JSONDIR="../exampleOutput/"
python batchEval.py $JSONDIR $MODELNAME "full" "val2014" $MAXPROP
#echo "final eval result in $LOGNAME"

