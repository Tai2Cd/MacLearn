�
 "serve*2.10.18�
8
dense/bias:VarHandleOp*
shape:@*
dtype0
K
dense/bias:/Read/ReadVariableOpReadVariableOpdense/bias:*
dtype0
>
dense/kernel:VarHandleOp*
shape
:	@*
dtype0
O
!dense/kernel:/Read/ReadVariableOpReadVariableOpdense/kernel:*
dtype0

NoOpNoOp
�
ConstConst*�
value�B��"�
�
layer_with_weights-0
layer-0
non_trainable_variables

layers
	variables
trainable_variables
	keras_api
v
non_trainable_variables

layers
		variables

trainable_variables
	keras_api

kernel
bias
 
 
 
 
 
 
 
 
 
 
[Y
VARIABLE_VALUEdense/kernel::06layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense/bias::04layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0
*
saver_filenamePlaceholder*
dtype0
8
Const_1Const*
valueB B^s3://.**
dtype0
9
RegexFullMatchRegexFullMatchsaver_filenameConst_1
5
Const_2Const*
valueB B.part*
dtype0
:
Const_3Const*
valueB B
_temp/part*
dtype0
;
SelectSelectRegexFullMatchConst_2Const_3*
T0
9

StringJoin
StringJoinsaver_filenameSelect*
N
4

num_shardsConst*
value	B :*
dtype0
1
Const_4Const*
value	B : *
dtype0
C
ShardedFilenameShardedFilename
StringJoinConst_4
num_shards
�
SaveV2/tensor_namesConst*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0
H
SaveV2/shape_and_slicesConst*
valueBB B B *
dtype0
�
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices!dense/kernel:/Read/ReadVariableOpdense/bias:/Read/ReadVariableOpConst*
dtypes
2
.
Const_5Const*
valueB *
dtype0
Q
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename*
T0*
N
Y
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixesConst_5
B
IdentityIdentitysaver_filename^MergeV2Checkpoints*
T0
�
RestoreV2/tensor_namesConst*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0
K
RestoreV2/shape_and_slicesConst*
valueBB B B *
dtype0
m
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices*
dtypes
2
*

Identity_1Identity	RestoreV2*
T0
L
AssignVariableOpAssignVariableOpdense/kernel:
Identity_1*
dtype0
,

Identity_2IdentityRestoreV2:1*
T0
L
AssignVariableOp_1AssignVariableOpdense/bias:
Identity_2*
dtype0

NoOp_1NoOp
`

Identity_3Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^NoOp_1*
T0 "�	,
saver_filename:0
Identity:0
Identity_38"
saved_model_main_op :�
�
layer_with_weights-0
layer-0
non_trainable_variables

layers
	variables
trainable_variables
	keras_api"
_tf_keras_layer
�
non_trainable_variables

layers
		variables

trainable_variables
	keras_api

kernel
bias"
_tf_keras_layer
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
:	@2dense/kernel
:@2
dense/bias