#!/bin/bash

echo $1
if [ "$1" == "TT_NN" ]; then
  echo "./apps/TT_NN/TTNN_model1.pth"
fi

if [ "$1" == "TT_FUNC" ]; then
  echo "./apps/TT_FUNC/TTFUNC_model1.pth"
  echo "./apps/TT_FUNC/TTFUNC_model2.pth"
  echo "./apps/TT_FUNC/TTFUNC_model3.pth"
  echo "./apps/TT_FUNC/TTFUNC_model4.pth"
  echo "./apps/TT_FUNC/TTFUNC_model5.pth"
  echo "./apps/TT_FUNC/TTFUNC_model6.pth"
  echo "./apps/TT_FUNC/TTFUNC_model7.pth"
  echo "./apps/TT_FUNC/TTFUNC_model8.pth"
fi

if [ $1 == "TT_DQN" ]; then
  echo "./apps/TT_DQN/TTDQN_model1.pth"
fi

if [ $1 == "ALL" ]; then
  echo "./apps/TT_NN/TTNN_model1.pth"
  echo "./apps/TT_FUNC/TTFUNC_model1.pth"
  echo "./apps/TT_FUNC/TTFUNC_model2.pth"
  echo "./apps/TT_FUNC/TTFUNC_model3.pth"
  echo "./apps/TT_FUNC/TTFUNC_model4.pth"
  echo "./apps/TT_FUNC/TTFUNC_model5.pth"
  echo "./apps/TT_FUNC/TTFUNC_model6.pth"
  echo "./apps/TT_FUNC/TTFUNC_model7.pth"
  echo "./apps/TT_FUNC/TTFUNC_model8.pth"
  echo "./apps/TT_DQN/TTDQN_model1.pth"
fi
