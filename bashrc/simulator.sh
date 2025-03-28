#!/bin/bash 
HOME="../"
SRC="$HOME/src"
RESULT="$HOME/result"
EXEC_PATH="$SRC/run.py"
SIMUL_LEN=48 


MODE_NAMETAG=("BASE" "GreenRec")

CARBON_NAMETAG=("NESO_Increase" "NESO_Decrease" "NESO_Stable")
CARBON_TRACE=(3478 3629 3430) #NESO_Increase, NESO_Decrease, NESO_Stable

WORKLOAD_NAMETAG=("movieLens" "Twitch")
WORKLOAD_TRACE=(40 10)

CONST_NAMETAG=("High" "Mid" "Low" "None")
MOVIELENS_CONST=(0.437424426 0.413123069 0.388821712 0) #Const_High, Const_Mid, Const_Low
TWITCH_CONST=(0.561575905 0.530377244 0.499178582 0) #Const_High, Const_Mid, Const_Low


#Arguments
#$1: simulation mode (0: BASE, 1: GreenRec)
#$2: workload selection (0: MovieLens, 1: Twitch)
#$3: carbon trace selection (0: Increase, 1: Decrease, 2: Stable)
#$4: accuracy constraint (0: High, 1: Mid, 2: Low 3: None)
#$5: refine parameter


#setup parameters
RQ_ST=${WORKLOAD_TRACE[$2]}
RQ_ET=`expr $RQ_ST \+ $SIMUL_LEN`

CI_ST=${CARBON_TRACE[$3]}
CI_ET=`expr $CI_ST \+ $SIMUL_LEN`

if [ $2 -eq 0 ]; then 
      CONST=${MOVIELENS_CONST[$4]}
else
      CONST=${TWITCH_CONST[$4]}
fi

DELTA=$5


#run simulator
CUDA_VISIBLE_DEVICES=1 python3 $EXEC_PATH -mode $1 -trace $2 -const $CONST -delta $DELTA\
      -rq_st $RQ_ST -rq_et $RQ_ET -ci_st $CI_ST -ci_et $CI_ET \
      > $RESULT/${MODE_NAMETAG[$1]}_${WORKLOAD_NAMETAG[$2]}_${CARBON_NAMETAG[$3]}_${CONST_NAMETAG[$4]}_$DELTA