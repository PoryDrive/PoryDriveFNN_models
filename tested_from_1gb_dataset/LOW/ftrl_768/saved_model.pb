£	
ë
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.02v2.6.0-rc2-32-g919f693420e8
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
À*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
À*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:À*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	À`*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	À`*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:`*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`0*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:`0*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:0*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:0*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
f
	Ftrl/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Ftrl/iter
_
Ftrl/iter/Read/ReadVariableOpReadVariableOp	Ftrl/iter*
_output_shapes
: *
dtype0	
f
	Ftrl/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	Ftrl/beta
_
Ftrl/beta/Read/ReadVariableOpReadVariableOp	Ftrl/beta*
_output_shapes
: *
dtype0
h

Ftrl/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Ftrl/decay
a
Ftrl/decay/Read/ReadVariableOpReadVariableOp
Ftrl/decay*
_output_shapes
: *
dtype0

Ftrl/l1_regularization_strengthVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Ftrl/l1_regularization_strength

3Ftrl/l1_regularization_strength/Read/ReadVariableOpReadVariableOpFtrl/l1_regularization_strength*
_output_shapes
: *
dtype0

Ftrl/l2_regularization_strengthVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Ftrl/l2_regularization_strength

3Ftrl/l2_regularization_strength/Read/ReadVariableOpReadVariableOpFtrl/l2_regularization_strength*
_output_shapes
: *
dtype0
x
Ftrl/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameFtrl/learning_rate
q
&Ftrl/learning_rate/Read/ReadVariableOpReadVariableOpFtrl/learning_rate*
_output_shapes
: *
dtype0

Ftrl/learning_rate_powerVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameFtrl/learning_rate_power
}
,Ftrl/learning_rate_power/Read/ReadVariableOpReadVariableOpFtrl/learning_rate_power*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Ftrl/dense/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*.
shared_nameFtrl/dense/kernel/accumulator

1Ftrl/dense/kernel/accumulator/Read/ReadVariableOpReadVariableOpFtrl/dense/kernel/accumulator*
_output_shapes
:	*
dtype0

Ftrl/dense/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameFtrl/dense/bias/accumulator

/Ftrl/dense/bias/accumulator/Read/ReadVariableOpReadVariableOpFtrl/dense/bias/accumulator*
_output_shapes	
:*
dtype0

Ftrl/dense_1/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*0
shared_name!Ftrl/dense_1/kernel/accumulator

3Ftrl/dense_1/kernel/accumulator/Read/ReadVariableOpReadVariableOpFtrl/dense_1/kernel/accumulator* 
_output_shapes
:
*
dtype0

Ftrl/dense_1/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameFtrl/dense_1/bias/accumulator

1Ftrl/dense_1/bias/accumulator/Read/ReadVariableOpReadVariableOpFtrl/dense_1/bias/accumulator*
_output_shapes	
:*
dtype0

Ftrl/dense_2/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
À*0
shared_name!Ftrl/dense_2/kernel/accumulator

3Ftrl/dense_2/kernel/accumulator/Read/ReadVariableOpReadVariableOpFtrl/dense_2/kernel/accumulator* 
_output_shapes
:
À*
dtype0

Ftrl/dense_2/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*.
shared_nameFtrl/dense_2/bias/accumulator

1Ftrl/dense_2/bias/accumulator/Read/ReadVariableOpReadVariableOpFtrl/dense_2/bias/accumulator*
_output_shapes	
:À*
dtype0

Ftrl/dense_3/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	À`*0
shared_name!Ftrl/dense_3/kernel/accumulator

3Ftrl/dense_3/kernel/accumulator/Read/ReadVariableOpReadVariableOpFtrl/dense_3/kernel/accumulator*
_output_shapes
:	À`*
dtype0

Ftrl/dense_3/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*.
shared_nameFtrl/dense_3/bias/accumulator

1Ftrl/dense_3/bias/accumulator/Read/ReadVariableOpReadVariableOpFtrl/dense_3/bias/accumulator*
_output_shapes
:`*
dtype0

Ftrl/dense_4/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`0*0
shared_name!Ftrl/dense_4/kernel/accumulator

3Ftrl/dense_4/kernel/accumulator/Read/ReadVariableOpReadVariableOpFtrl/dense_4/kernel/accumulator*
_output_shapes

:`0*
dtype0

Ftrl/dense_4/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*.
shared_nameFtrl/dense_4/bias/accumulator

1Ftrl/dense_4/bias/accumulator/Read/ReadVariableOpReadVariableOpFtrl/dense_4/bias/accumulator*
_output_shapes
:0*
dtype0

Ftrl/dense_5/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0*0
shared_name!Ftrl/dense_5/kernel/accumulator

3Ftrl/dense_5/kernel/accumulator/Read/ReadVariableOpReadVariableOpFtrl/dense_5/kernel/accumulator*
_output_shapes

:0*
dtype0

Ftrl/dense_5/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameFtrl/dense_5/bias/accumulator

1Ftrl/dense_5/bias/accumulator/Read/ReadVariableOpReadVariableOpFtrl/dense_5/bias/accumulator*
_output_shapes
:*
dtype0

Ftrl/dense/kernel/linearVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameFtrl/dense/kernel/linear

,Ftrl/dense/kernel/linear/Read/ReadVariableOpReadVariableOpFtrl/dense/kernel/linear*
_output_shapes
:	*
dtype0

Ftrl/dense/bias/linearVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameFtrl/dense/bias/linear
~
*Ftrl/dense/bias/linear/Read/ReadVariableOpReadVariableOpFtrl/dense/bias/linear*
_output_shapes	
:*
dtype0

Ftrl/dense_1/kernel/linearVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameFtrl/dense_1/kernel/linear

.Ftrl/dense_1/kernel/linear/Read/ReadVariableOpReadVariableOpFtrl/dense_1/kernel/linear* 
_output_shapes
:
*
dtype0

Ftrl/dense_1/bias/linearVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameFtrl/dense_1/bias/linear

,Ftrl/dense_1/bias/linear/Read/ReadVariableOpReadVariableOpFtrl/dense_1/bias/linear*
_output_shapes	
:*
dtype0

Ftrl/dense_2/kernel/linearVarHandleOp*
_output_shapes
: *
dtype0*
shape:
À*+
shared_nameFtrl/dense_2/kernel/linear

.Ftrl/dense_2/kernel/linear/Read/ReadVariableOpReadVariableOpFtrl/dense_2/kernel/linear* 
_output_shapes
:
À*
dtype0

Ftrl/dense_2/bias/linearVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*)
shared_nameFtrl/dense_2/bias/linear

,Ftrl/dense_2/bias/linear/Read/ReadVariableOpReadVariableOpFtrl/dense_2/bias/linear*
_output_shapes	
:À*
dtype0

Ftrl/dense_3/kernel/linearVarHandleOp*
_output_shapes
: *
dtype0*
shape:	À`*+
shared_nameFtrl/dense_3/kernel/linear

.Ftrl/dense_3/kernel/linear/Read/ReadVariableOpReadVariableOpFtrl/dense_3/kernel/linear*
_output_shapes
:	À`*
dtype0

Ftrl/dense_3/bias/linearVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*)
shared_nameFtrl/dense_3/bias/linear

,Ftrl/dense_3/bias/linear/Read/ReadVariableOpReadVariableOpFtrl/dense_3/bias/linear*
_output_shapes
:`*
dtype0

Ftrl/dense_4/kernel/linearVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`0*+
shared_nameFtrl/dense_4/kernel/linear

.Ftrl/dense_4/kernel/linear/Read/ReadVariableOpReadVariableOpFtrl/dense_4/kernel/linear*
_output_shapes

:`0*
dtype0

Ftrl/dense_4/bias/linearVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*)
shared_nameFtrl/dense_4/bias/linear

,Ftrl/dense_4/bias/linear/Read/ReadVariableOpReadVariableOpFtrl/dense_4/bias/linear*
_output_shapes
:0*
dtype0

Ftrl/dense_5/kernel/linearVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0*+
shared_nameFtrl/dense_5/kernel/linear

.Ftrl/dense_5/kernel/linear/Read/ReadVariableOpReadVariableOpFtrl/dense_5/kernel/linear*
_output_shapes

:0*
dtype0

Ftrl/dense_5/bias/linearVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameFtrl/dense_5/bias/linear

,Ftrl/dense_5/bias/linear/Read/ReadVariableOpReadVariableOpFtrl/dense_5/bias/linear*
_output_shapes
:*
dtype0

NoOpNoOp
D
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÓC
valueÉCBÆC B¿C
Û
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
h

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
h

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api

1iter
2beta
	3decay
4l1_regularization_strength
5l2_regularization_strength
6learning_rate
7learning_rate_poweraccumulator`accumulatoraaccumulatorbaccumulatorcaccumulatordaccumulatoreaccumulatorf accumulatorg%accumulatorh&accumulatori+accumulatorj,accumulatorklinearllinearmlinearnlinearolinearplinearqlinearr linears%lineart&linearu+linearv,linearw
V
0
1
2
3
4
5
6
 7
%8
&9
+10
,11
V
0
1
2
3
4
5
6
 7
%8
&9
+10
,11
 
­
8layer_metrics
	variables
9metrics
	trainable_variables

:layers
;non_trainable_variables

regularization_losses
<layer_regularization_losses
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
=layer_metrics
	variables
>metrics
trainable_variables

?layers
@non_trainable_variables
regularization_losses
Alayer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
Blayer_metrics
	variables
Cmetrics
trainable_variables

Dlayers
Enon_trainable_variables
regularization_losses
Flayer_regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
Glayer_metrics
	variables
Hmetrics
trainable_variables

Ilayers
Jnon_trainable_variables
regularization_losses
Klayer_regularization_losses
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
­
Llayer_metrics
!	variables
Mmetrics
"trainable_variables

Nlayers
Onon_trainable_variables
#regularization_losses
Player_regularization_losses
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1

%0
&1
 
­
Qlayer_metrics
'	variables
Rmetrics
(trainable_variables

Slayers
Tnon_trainable_variables
)regularization_losses
Ulayer_regularization_losses
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1

+0
,1
 
­
Vlayer_metrics
-	variables
Wmetrics
.trainable_variables

Xlayers
Ynon_trainable_variables
/regularization_losses
Zlayer_regularization_losses
HF
VARIABLE_VALUE	Ftrl/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUE	Ftrl/beta)optimizer/beta/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Ftrl/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEFtrl/l1_regularization_strength?optimizer/l1_regularization_strength/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEFtrl/l2_regularization_strength?optimizer/l2_regularization_strength/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEFtrl/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEFtrl/learning_rate_power8optimizer/learning_rate_power/.ATTRIBUTES/VARIABLE_VALUE
 

[0
*
0
1
2
3
4
5
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
 
 
4
	\total
	]count
^	variables
_	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

\0
]1

^	variables

VARIABLE_VALUEFtrl/dense/kernel/accumulator\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense/bias/accumulatorZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_1/kernel/accumulator\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_1/bias/accumulatorZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_2/kernel/accumulator\layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_2/bias/accumulatorZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_3/kernel/accumulator\layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_3/bias/accumulatorZlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_4/kernel/accumulator\layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_4/bias/accumulatorZlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_5/kernel/accumulator\layer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_5/bias/accumulatorZlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense/kernel/linearWlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense/bias/linearUlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_1/kernel/linearWlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_1/bias/linearUlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_2/kernel/linearWlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_2/bias/linearUlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_3/kernel/linearWlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_3/bias/linearUlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_4/kernel/linearWlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_4/bias/linearUlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_5/kernel/linearWlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFtrl/dense_5/bias/linearUlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUE
~
serving_default_dense_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
þ
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_inputdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_647868
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ø
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpFtrl/iter/Read/ReadVariableOpFtrl/beta/Read/ReadVariableOpFtrl/decay/Read/ReadVariableOp3Ftrl/l1_regularization_strength/Read/ReadVariableOp3Ftrl/l2_regularization_strength/Read/ReadVariableOp&Ftrl/learning_rate/Read/ReadVariableOp,Ftrl/learning_rate_power/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1Ftrl/dense/kernel/accumulator/Read/ReadVariableOp/Ftrl/dense/bias/accumulator/Read/ReadVariableOp3Ftrl/dense_1/kernel/accumulator/Read/ReadVariableOp1Ftrl/dense_1/bias/accumulator/Read/ReadVariableOp3Ftrl/dense_2/kernel/accumulator/Read/ReadVariableOp1Ftrl/dense_2/bias/accumulator/Read/ReadVariableOp3Ftrl/dense_3/kernel/accumulator/Read/ReadVariableOp1Ftrl/dense_3/bias/accumulator/Read/ReadVariableOp3Ftrl/dense_4/kernel/accumulator/Read/ReadVariableOp1Ftrl/dense_4/bias/accumulator/Read/ReadVariableOp3Ftrl/dense_5/kernel/accumulator/Read/ReadVariableOp1Ftrl/dense_5/bias/accumulator/Read/ReadVariableOp,Ftrl/dense/kernel/linear/Read/ReadVariableOp*Ftrl/dense/bias/linear/Read/ReadVariableOp.Ftrl/dense_1/kernel/linear/Read/ReadVariableOp,Ftrl/dense_1/bias/linear/Read/ReadVariableOp.Ftrl/dense_2/kernel/linear/Read/ReadVariableOp,Ftrl/dense_2/bias/linear/Read/ReadVariableOp.Ftrl/dense_3/kernel/linear/Read/ReadVariableOp,Ftrl/dense_3/bias/linear/Read/ReadVariableOp.Ftrl/dense_4/kernel/linear/Read/ReadVariableOp,Ftrl/dense_4/bias/linear/Read/ReadVariableOp.Ftrl/dense_5/kernel/linear/Read/ReadVariableOp,Ftrl/dense_5/bias/linear/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_648296
Ï

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias	Ftrl/iter	Ftrl/beta
Ftrl/decayFtrl/l1_regularization_strengthFtrl/l2_regularization_strengthFtrl/learning_rateFtrl/learning_rate_powertotalcountFtrl/dense/kernel/accumulatorFtrl/dense/bias/accumulatorFtrl/dense_1/kernel/accumulatorFtrl/dense_1/bias/accumulatorFtrl/dense_2/kernel/accumulatorFtrl/dense_2/bias/accumulatorFtrl/dense_3/kernel/accumulatorFtrl/dense_3/bias/accumulatorFtrl/dense_4/kernel/accumulatorFtrl/dense_4/bias/accumulatorFtrl/dense_5/kernel/accumulatorFtrl/dense_5/bias/accumulatorFtrl/dense/kernel/linearFtrl/dense/bias/linearFtrl/dense_1/kernel/linearFtrl/dense_1/bias/linearFtrl/dense_2/kernel/linearFtrl/dense_2/bias/linearFtrl/dense_3/kernel/linearFtrl/dense_3/bias/linearFtrl/dense_4/kernel/linearFtrl/dense_4/bias/linearFtrl/dense_5/kernel/linearFtrl/dense_5/bias/linear*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_648441ôÓ
ï

(__inference_dense_4_layer_call_fn_648118

inputs
unknown:`0
	unknown_0:0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_6475272
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
â!
¹
F__inference_sequential_layer_call_and_return_conditional_losses_647793
dense_input
dense_647762:	
dense_647764:	"
dense_1_647767:

dense_1_647769:	"
dense_2_647772:
À
dense_2_647774:	À!
dense_3_647777:	À`
dense_3_647779:` 
dense_4_647782:`0
dense_4_647784:0 
dense_5_647787:0
dense_5_647789:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_647762dense_647764*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6474592
dense/StatefulPartitionedCall°
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_647767dense_1_647769*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6474762!
dense_1/StatefulPartitionedCall²
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_647772dense_2_647774*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_6474932!
dense_2/StatefulPartitionedCall±
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_647777dense_3_647779*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_6475102!
dense_3/StatefulPartitionedCall±
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_647782dense_4_647784*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_6475272!
dense_4/StatefulPartitionedCall±
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_647787dense_5_647789*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_6475442!
dense_5/StatefulPartitionedCall
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input

®
$__inference_signature_wrapper_647868
dense_input
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:
À
	unknown_4:	À
	unknown_5:	À`
	unknown_6:`
	unknown_7:`0
	unknown_8:0
	unknown_9:0

unknown_10:
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_6474412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
û

õ
C__inference_dense_3_layer_call_and_return_conditional_losses_647510

inputs1
matmul_readvariableop_resource:	À`-
biasadd_readvariableop_resource:`
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	À`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
â!
¹
F__inference_sequential_layer_call_and_return_conditional_losses_647827
dense_input
dense_647796:	
dense_647798:	"
dense_1_647801:

dense_1_647803:	"
dense_2_647806:
À
dense_2_647808:	À!
dense_3_647811:	À`
dense_3_647813:` 
dense_4_647816:`0
dense_4_647818:0 
dense_5_647821:0
dense_5_647823:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_647796dense_647798*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6474592
dense/StatefulPartitionedCall°
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_647801dense_1_647803*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6474762!
dense_1/StatefulPartitionedCall²
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_647806dense_2_647808*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_6474932!
dense_2/StatefulPartitionedCall±
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_647811dense_3_647813*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_6475102!
dense_3/StatefulPartitionedCall±
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_647816dense_4_647818*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_6475272!
dense_4/StatefulPartitionedCall±
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_647821dense_5_647823*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_6475442!
dense_5/StatefulPartitionedCall
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
ý

ô
A__inference_dense_layer_call_and_return_conditional_losses_647459

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanhd
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó!
´
F__inference_sequential_layer_call_and_return_conditional_losses_647551

inputs
dense_647460:	
dense_647462:	"
dense_1_647477:

dense_1_647479:	"
dense_2_647494:
À
dense_2_647496:	À!
dense_3_647511:	À`
dense_3_647513:` 
dense_4_647528:`0
dense_4_647530:0 
dense_5_647545:0
dense_5_647547:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_647460dense_647462*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6474592
dense/StatefulPartitionedCall°
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_647477dense_1_647479*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6474762!
dense_1/StatefulPartitionedCall²
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_647494dense_2_647496*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_6474932!
dense_2/StatefulPartitionedCall±
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_647511dense_3_647513*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_6475102!
dense_3/StatefulPartitionedCall±
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_647528dense_4_647530*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_6475272!
dense_4/StatefulPartitionedCall±
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_647545dense_5_647547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_6475442!
dense_5/StatefulPartitionedCall
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

÷
C__inference_dense_2_layer_call_and_return_conditional_losses_647493

inputs2
matmul_readvariableop_resource:
À.
biasadd_readvariableop_resource:	À
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:À*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
Tanhd
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷

ô
C__inference_dense_4_layer_call_and_return_conditional_losses_648109

inputs0
matmul_readvariableop_resource:`0-
biasadd_readvariableop_resource:0
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
9
	
F__inference_sequential_layer_call_and_return_conditional_losses_647960

inputs7
$dense_matmul_readvariableop_resource:	4
%dense_biasadd_readvariableop_resource:	:
&dense_1_matmul_readvariableop_resource:
6
'dense_1_biasadd_readvariableop_resource:	:
&dense_2_matmul_readvariableop_resource:
À6
'dense_2_biasadd_readvariableop_resource:	À9
&dense_3_matmul_readvariableop_resource:	À`5
'dense_3_biasadd_readvariableop_resource:`8
&dense_4_matmul_readvariableop_resource:`05
'dense_4_biasadd_readvariableop_resource:08
&dense_5_matmul_readvariableop_resource:05
'dense_5_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddk

dense/TanhTanhdense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Tanh§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Tanh:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddq
dense_1/TanhTanhdense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Tanh§
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Tanh:y:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
dense_2/MatMul¥
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:À*
dtype02 
dense_2/BiasAdd/ReadVariableOp¢
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
dense_2/BiasAddq
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
dense_2/Tanh¦
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	À`*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMuldense_2/Tanh:y:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
dense_3/MatMul¤
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02 
dense_3/BiasAdd/ReadVariableOp¡
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
dense_3/BiasAddp
dense_3/TanhTanhdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
dense_3/Tanh¥
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:`0*
dtype02
dense_4/MatMul/ReadVariableOp
dense_4/MatMulMatMuldense_3/Tanh:y:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
dense_4/MatMul¤
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02 
dense_4/BiasAdd/ReadVariableOp¡
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
dense_4/BiasAddp
dense_4/TanhTanhdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
dense_4/Tanh¥
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:0*
dtype02
dense_5/MatMul/ReadVariableOp
dense_5/MatMulMatMuldense_4/Tanh:y:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/MatMul¤
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp¡
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/BiasAddp
dense_5/TanhTanhdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/Tanhk
IdentityIdentitydense_5/Tanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÐ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û

õ
C__inference_dense_3_layer_call_and_return_conditional_losses_648089

inputs1
matmul_readvariableop_resource:	À`-
biasadd_readvariableop_resource:`
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	À`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
÷

ô
C__inference_dense_5_layer_call_and_return_conditional_losses_648129

inputs0
matmul_readvariableop_resource:0-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
9
	
F__inference_sequential_layer_call_and_return_conditional_losses_647914

inputs7
$dense_matmul_readvariableop_resource:	4
%dense_biasadd_readvariableop_resource:	:
&dense_1_matmul_readvariableop_resource:
6
'dense_1_biasadd_readvariableop_resource:	:
&dense_2_matmul_readvariableop_resource:
À6
'dense_2_biasadd_readvariableop_resource:	À9
&dense_3_matmul_readvariableop_resource:	À`5
'dense_3_biasadd_readvariableop_resource:`8
&dense_4_matmul_readvariableop_resource:`05
'dense_4_biasadd_readvariableop_resource:08
&dense_5_matmul_readvariableop_resource:05
'dense_5_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddk

dense/TanhTanhdense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Tanh§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Tanh:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddq
dense_1/TanhTanhdense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Tanh§
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Tanh:y:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
dense_2/MatMul¥
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:À*
dtype02 
dense_2/BiasAdd/ReadVariableOp¢
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
dense_2/BiasAddq
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
dense_2/Tanh¦
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	À`*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMuldense_2/Tanh:y:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
dense_3/MatMul¤
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02 
dense_3/BiasAdd/ReadVariableOp¡
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
dense_3/BiasAddp
dense_3/TanhTanhdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
dense_3/Tanh¥
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:`0*
dtype02
dense_4/MatMul/ReadVariableOp
dense_4/MatMulMatMuldense_3/Tanh:y:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
dense_4/MatMul¤
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02 
dense_4/BiasAdd/ReadVariableOp¡
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
dense_4/BiasAddp
dense_4/TanhTanhdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
dense_4/Tanh¥
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:0*
dtype02
dense_5/MatMul/ReadVariableOp
dense_5/MatMulMatMuldense_4/Tanh:y:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/MatMul¤
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp¡
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/BiasAddp
dense_5/TanhTanhdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/Tanhk
IdentityIdentitydense_5/Tanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÐ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
íF
ú

!__inference__wrapped_model_647441
dense_inputB
/sequential_dense_matmul_readvariableop_resource:	?
0sequential_dense_biasadd_readvariableop_resource:	E
1sequential_dense_1_matmul_readvariableop_resource:
A
2sequential_dense_1_biasadd_readvariableop_resource:	E
1sequential_dense_2_matmul_readvariableop_resource:
ÀA
2sequential_dense_2_biasadd_readvariableop_resource:	ÀD
1sequential_dense_3_matmul_readvariableop_resource:	À`@
2sequential_dense_3_biasadd_readvariableop_resource:`C
1sequential_dense_4_matmul_readvariableop_resource:`0@
2sequential_dense_4_biasadd_readvariableop_resource:0C
1sequential_dense_5_matmul_readvariableop_resource:0@
2sequential_dense_5_biasadd_readvariableop_resource:
identity¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢(sequential/dense_1/MatMul/ReadVariableOp¢)sequential/dense_2/BiasAdd/ReadVariableOp¢(sequential/dense_2/MatMul/ReadVariableOp¢)sequential/dense_3/BiasAdd/ReadVariableOp¢(sequential/dense_3/MatMul/ReadVariableOp¢)sequential/dense_4/BiasAdd/ReadVariableOp¢(sequential/dense_4/MatMul/ReadVariableOp¢)sequential/dense_5/BiasAdd/ReadVariableOp¢(sequential/dense_5/MatMul/ReadVariableOpÁ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02(
&sequential/dense/MatMul/ReadVariableOp¬
sequential/dense/MatMulMatMuldense_input.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/MatMulÀ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpÆ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/BiasAdd
sequential/dense/TanhTanh!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/TanhÈ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpÀ
sequential/dense_1/MatMulMatMulsequential/dense/Tanh:y:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/MatMulÆ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpÎ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/BiasAdd
sequential/dense_1/TanhTanh#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/TanhÈ
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOpÂ
sequential/dense_2/MatMulMatMulsequential/dense_1/Tanh:y:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
sequential/dense_2/MatMulÆ
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:À*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOpÎ
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
sequential/dense_2/BiasAdd
sequential/dense_2/TanhTanh#sequential/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
sequential/dense_2/TanhÇ
(sequential/dense_3/MatMul/ReadVariableOpReadVariableOp1sequential_dense_3_matmul_readvariableop_resource*
_output_shapes
:	À`*
dtype02*
(sequential/dense_3/MatMul/ReadVariableOpÁ
sequential/dense_3/MatMulMatMulsequential/dense_2/Tanh:y:00sequential/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
sequential/dense_3/MatMulÅ
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02+
)sequential/dense_3/BiasAdd/ReadVariableOpÍ
sequential/dense_3/BiasAddBiasAdd#sequential/dense_3/MatMul:product:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
sequential/dense_3/BiasAdd
sequential/dense_3/TanhTanh#sequential/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
sequential/dense_3/TanhÆ
(sequential/dense_4/MatMul/ReadVariableOpReadVariableOp1sequential_dense_4_matmul_readvariableop_resource*
_output_shapes

:`0*
dtype02*
(sequential/dense_4/MatMul/ReadVariableOpÁ
sequential/dense_4/MatMulMatMulsequential/dense_3/Tanh:y:00sequential/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
sequential/dense_4/MatMulÅ
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02+
)sequential/dense_4/BiasAdd/ReadVariableOpÍ
sequential/dense_4/BiasAddBiasAdd#sequential/dense_4/MatMul:product:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
sequential/dense_4/BiasAdd
sequential/dense_4/TanhTanh#sequential/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
sequential/dense_4/TanhÆ
(sequential/dense_5/MatMul/ReadVariableOpReadVariableOp1sequential_dense_5_matmul_readvariableop_resource*
_output_shapes

:0*
dtype02*
(sequential/dense_5/MatMul/ReadVariableOpÁ
sequential/dense_5/MatMulMatMulsequential/dense_4/Tanh:y:00sequential/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_5/MatMulÅ
)sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_5/BiasAdd/ReadVariableOpÍ
sequential/dense_5/BiasAddBiasAdd#sequential/dense_5/MatMul:product:01sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_5/BiasAdd
sequential/dense_5/TanhTanh#sequential/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_5/Tanhv
IdentityIdentitysequential/dense_5/Tanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÔ
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp)^sequential/dense_3/MatMul/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp)^sequential/dense_4/MatMul/ReadVariableOp*^sequential/dense_5/BiasAdd/ReadVariableOp)^sequential/dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2V
)sequential/dense_3/BiasAdd/ReadVariableOp)sequential/dense_3/BiasAdd/ReadVariableOp2T
(sequential/dense_3/MatMul/ReadVariableOp(sequential/dense_3/MatMul/ReadVariableOp2V
)sequential/dense_4/BiasAdd/ReadVariableOp)sequential/dense_4/BiasAdd/ReadVariableOp2T
(sequential/dense_4/MatMul/ReadVariableOp(sequential/dense_4/MatMul/ReadVariableOp2V
)sequential/dense_5/BiasAdd/ReadVariableOp)sequential/dense_5/BiasAdd/ReadVariableOp2T
(sequential/dense_5/MatMul/ReadVariableOp(sequential/dense_5/MatMul/ReadVariableOp:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
ï

&__inference_dense_layer_call_fn_648038

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6474592
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þÅ
Ô
"__inference__traced_restore_648441
file_prefix0
assignvariableop_dense_kernel:	,
assignvariableop_1_dense_bias:	5
!assignvariableop_2_dense_1_kernel:
.
assignvariableop_3_dense_1_bias:	5
!assignvariableop_4_dense_2_kernel:
À.
assignvariableop_5_dense_2_bias:	À4
!assignvariableop_6_dense_3_kernel:	À`-
assignvariableop_7_dense_3_bias:`3
!assignvariableop_8_dense_4_kernel:`0-
assignvariableop_9_dense_4_bias:04
"assignvariableop_10_dense_5_kernel:0.
 assignvariableop_11_dense_5_bias:'
assignvariableop_12_ftrl_iter:	 '
assignvariableop_13_ftrl_beta: (
assignvariableop_14_ftrl_decay: =
3assignvariableop_15_ftrl_l1_regularization_strength: =
3assignvariableop_16_ftrl_l2_regularization_strength: 0
&assignvariableop_17_ftrl_learning_rate: 6
,assignvariableop_18_ftrl_learning_rate_power: #
assignvariableop_19_total: #
assignvariableop_20_count: D
1assignvariableop_21_ftrl_dense_kernel_accumulator:	>
/assignvariableop_22_ftrl_dense_bias_accumulator:	G
3assignvariableop_23_ftrl_dense_1_kernel_accumulator:
@
1assignvariableop_24_ftrl_dense_1_bias_accumulator:	G
3assignvariableop_25_ftrl_dense_2_kernel_accumulator:
À@
1assignvariableop_26_ftrl_dense_2_bias_accumulator:	ÀF
3assignvariableop_27_ftrl_dense_3_kernel_accumulator:	À`?
1assignvariableop_28_ftrl_dense_3_bias_accumulator:`E
3assignvariableop_29_ftrl_dense_4_kernel_accumulator:`0?
1assignvariableop_30_ftrl_dense_4_bias_accumulator:0E
3assignvariableop_31_ftrl_dense_5_kernel_accumulator:0?
1assignvariableop_32_ftrl_dense_5_bias_accumulator:?
,assignvariableop_33_ftrl_dense_kernel_linear:	9
*assignvariableop_34_ftrl_dense_bias_linear:	B
.assignvariableop_35_ftrl_dense_1_kernel_linear:
;
,assignvariableop_36_ftrl_dense_1_bias_linear:	B
.assignvariableop_37_ftrl_dense_2_kernel_linear:
À;
,assignvariableop_38_ftrl_dense_2_bias_linear:	ÀA
.assignvariableop_39_ftrl_dense_3_kernel_linear:	À`:
,assignvariableop_40_ftrl_dense_3_bias_linear:`@
.assignvariableop_41_ftrl_dense_4_kernel_linear:`0:
,assignvariableop_42_ftrl_dense_4_bias_linear:0@
.assignvariableop_43_ftrl_dense_5_kernel_linear:0:
,assignvariableop_44_ftrl_dense_5_bias_linear:
identity_46¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*¡
valueB.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/beta/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB?optimizer/l1_regularization_strength/.ATTRIBUTES/VARIABLE_VALUEB?optimizer/l2_regularization_strength/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB8optimizer/learning_rate_power/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesê
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Î
_output_shapes»
¸::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¢
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¦
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¤
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¦
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¤
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¦
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¤
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¦
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¤
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ª
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¨
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12¥
AssignVariableOp_12AssignVariableOpassignvariableop_12_ftrl_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¥
AssignVariableOp_13AssignVariableOpassignvariableop_13_ftrl_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¦
AssignVariableOp_14AssignVariableOpassignvariableop_14_ftrl_decayIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15»
AssignVariableOp_15AssignVariableOp3assignvariableop_15_ftrl_l1_regularization_strengthIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16»
AssignVariableOp_16AssignVariableOp3assignvariableop_16_ftrl_l2_regularization_strengthIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17®
AssignVariableOp_17AssignVariableOp&assignvariableop_17_ftrl_learning_rateIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18´
AssignVariableOp_18AssignVariableOp,assignvariableop_18_ftrl_learning_rate_powerIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¡
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¡
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¹
AssignVariableOp_21AssignVariableOp1assignvariableop_21_ftrl_dense_kernel_accumulatorIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22·
AssignVariableOp_22AssignVariableOp/assignvariableop_22_ftrl_dense_bias_accumulatorIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23»
AssignVariableOp_23AssignVariableOp3assignvariableop_23_ftrl_dense_1_kernel_accumulatorIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¹
AssignVariableOp_24AssignVariableOp1assignvariableop_24_ftrl_dense_1_bias_accumulatorIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25»
AssignVariableOp_25AssignVariableOp3assignvariableop_25_ftrl_dense_2_kernel_accumulatorIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¹
AssignVariableOp_26AssignVariableOp1assignvariableop_26_ftrl_dense_2_bias_accumulatorIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27»
AssignVariableOp_27AssignVariableOp3assignvariableop_27_ftrl_dense_3_kernel_accumulatorIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¹
AssignVariableOp_28AssignVariableOp1assignvariableop_28_ftrl_dense_3_bias_accumulatorIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29»
AssignVariableOp_29AssignVariableOp3assignvariableop_29_ftrl_dense_4_kernel_accumulatorIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¹
AssignVariableOp_30AssignVariableOp1assignvariableop_30_ftrl_dense_4_bias_accumulatorIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31»
AssignVariableOp_31AssignVariableOp3assignvariableop_31_ftrl_dense_5_kernel_accumulatorIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¹
AssignVariableOp_32AssignVariableOp1assignvariableop_32_ftrl_dense_5_bias_accumulatorIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33´
AssignVariableOp_33AssignVariableOp,assignvariableop_33_ftrl_dense_kernel_linearIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34²
AssignVariableOp_34AssignVariableOp*assignvariableop_34_ftrl_dense_bias_linearIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¶
AssignVariableOp_35AssignVariableOp.assignvariableop_35_ftrl_dense_1_kernel_linearIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36´
AssignVariableOp_36AssignVariableOp,assignvariableop_36_ftrl_dense_1_bias_linearIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37¶
AssignVariableOp_37AssignVariableOp.assignvariableop_37_ftrl_dense_2_kernel_linearIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38´
AssignVariableOp_38AssignVariableOp,assignvariableop_38_ftrl_dense_2_bias_linearIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¶
AssignVariableOp_39AssignVariableOp.assignvariableop_39_ftrl_dense_3_kernel_linearIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40´
AssignVariableOp_40AssignVariableOp,assignvariableop_40_ftrl_dense_3_bias_linearIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41¶
AssignVariableOp_41AssignVariableOp.assignvariableop_41_ftrl_dense_4_kernel_linearIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42´
AssignVariableOp_42AssignVariableOp,assignvariableop_42_ftrl_dense_4_bias_linearIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43¶
AssignVariableOp_43AssignVariableOp.assignvariableop_43_ftrl_dense_5_kernel_linearIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44´
AssignVariableOp_44AssignVariableOp,assignvariableop_44_ftrl_dense_5_bias_linearIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¼
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45f
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_46¤
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ò

(__inference_dense_3_layer_call_fn_648098

inputs
unknown:	À`
	unknown_0:`
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_6475102
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs

÷
C__inference_dense_2_layer_call_and_return_conditional_losses_648069

inputs2
matmul_readvariableop_resource:
À.
biasadd_readvariableop_resource:	À
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:À*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
Tanhd
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷

ô
C__inference_dense_5_layer_call_and_return_conditional_losses_647544

inputs0
matmul_readvariableop_resource:0-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
÷

ô
C__inference_dense_4_layer_call_and_return_conditional_losses_647527

inputs0
matmul_readvariableop_resource:`0-
biasadd_readvariableop_resource:0
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
Ó!
´
F__inference_sequential_layer_call_and_return_conditional_losses_647703

inputs
dense_647672:	
dense_647674:	"
dense_1_647677:

dense_1_647679:	"
dense_2_647682:
À
dense_2_647684:	À!
dense_3_647687:	À`
dense_3_647689:` 
dense_4_647692:`0
dense_4_647694:0 
dense_5_647697:0
dense_5_647699:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_647672dense_647674*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6474592
dense/StatefulPartitionedCall°
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_647677dense_1_647679*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6474762!
dense_1/StatefulPartitionedCall²
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_647682dense_2_647684*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_6474932!
dense_2/StatefulPartitionedCall±
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_647687dense_3_647689*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_6475102!
dense_3/StatefulPartitionedCall±
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_647692dense_4_647694*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_6475272!
dense_4/StatefulPartitionedCall±
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_647697dense_5_647699*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_6475442!
dense_5/StatefulPartitionedCall
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

÷
C__inference_dense_1_layer_call_and_return_conditional_losses_648049

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanhd
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö

(__inference_dense_1_layer_call_fn_648058

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6474762
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
µ
+__inference_sequential_layer_call_fn_647759
dense_input
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:
À
	unknown_4:	À
	unknown_5:	À`
	unknown_6:`
	unknown_7:`0
	unknown_8:0
	unknown_9:0

unknown_10:
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_6477032
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
ö

(__inference_dense_2_layer_call_fn_648078

inputs
unknown:
À
	unknown_0:	À
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_6474932
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
°
+__inference_sequential_layer_call_fn_647989

inputs
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:
À
	unknown_4:	À
	unknown_5:	À`
	unknown_6:`
	unknown_7:`0
	unknown_8:0
	unknown_9:0

unknown_10:
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_6475512
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï

(__inference_dense_5_layer_call_fn_648138

inputs
unknown:0
	unknown_0:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_6475442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
³
µ
+__inference_sequential_layer_call_fn_647578
dense_input
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:
À
	unknown_4:	À
	unknown_5:	À`
	unknown_6:`
	unknown_7:`0
	unknown_8:0
	unknown_9:0

unknown_10:
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_6475512
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
¤
°
+__inference_sequential_layer_call_fn_648018

inputs
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:
À
	unknown_4:	À
	unknown_5:	À`
	unknown_6:`
	unknown_7:`0
	unknown_8:0
	unknown_9:0

unknown_10:
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_6477032
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

÷
C__inference_dense_1_layer_call_and_return_conditional_losses_647476

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanhd
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý

ô
A__inference_dense_layer_call_and_return_conditional_losses_648029

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanhd
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
øa
ë
__inference__traced_save_648296
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop(
$savev2_ftrl_iter_read_readvariableop	(
$savev2_ftrl_beta_read_readvariableop)
%savev2_ftrl_decay_read_readvariableop>
:savev2_ftrl_l1_regularization_strength_read_readvariableop>
:savev2_ftrl_l2_regularization_strength_read_readvariableop1
-savev2_ftrl_learning_rate_read_readvariableop7
3savev2_ftrl_learning_rate_power_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_ftrl_dense_kernel_accumulator_read_readvariableop:
6savev2_ftrl_dense_bias_accumulator_read_readvariableop>
:savev2_ftrl_dense_1_kernel_accumulator_read_readvariableop<
8savev2_ftrl_dense_1_bias_accumulator_read_readvariableop>
:savev2_ftrl_dense_2_kernel_accumulator_read_readvariableop<
8savev2_ftrl_dense_2_bias_accumulator_read_readvariableop>
:savev2_ftrl_dense_3_kernel_accumulator_read_readvariableop<
8savev2_ftrl_dense_3_bias_accumulator_read_readvariableop>
:savev2_ftrl_dense_4_kernel_accumulator_read_readvariableop<
8savev2_ftrl_dense_4_bias_accumulator_read_readvariableop>
:savev2_ftrl_dense_5_kernel_accumulator_read_readvariableop<
8savev2_ftrl_dense_5_bias_accumulator_read_readvariableop7
3savev2_ftrl_dense_kernel_linear_read_readvariableop5
1savev2_ftrl_dense_bias_linear_read_readvariableop9
5savev2_ftrl_dense_1_kernel_linear_read_readvariableop7
3savev2_ftrl_dense_1_bias_linear_read_readvariableop9
5savev2_ftrl_dense_2_kernel_linear_read_readvariableop7
3savev2_ftrl_dense_2_bias_linear_read_readvariableop9
5savev2_ftrl_dense_3_kernel_linear_read_readvariableop7
3savev2_ftrl_dense_3_bias_linear_read_readvariableop9
5savev2_ftrl_dense_4_kernel_linear_read_readvariableop7
3savev2_ftrl_dense_4_bias_linear_read_readvariableop9
5savev2_ftrl_dense_5_kernel_linear_read_readvariableop7
3savev2_ftrl_dense_5_bias_linear_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*¡
valueB.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/beta/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB?optimizer/l1_regularization_strength/.ATTRIBUTES/VARIABLE_VALUEB?optimizer/l2_regularization_strength/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB8optimizer/learning_rate_power/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/linear/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesä
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices²
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop$savev2_ftrl_iter_read_readvariableop$savev2_ftrl_beta_read_readvariableop%savev2_ftrl_decay_read_readvariableop:savev2_ftrl_l1_regularization_strength_read_readvariableop:savev2_ftrl_l2_regularization_strength_read_readvariableop-savev2_ftrl_learning_rate_read_readvariableop3savev2_ftrl_learning_rate_power_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_ftrl_dense_kernel_accumulator_read_readvariableop6savev2_ftrl_dense_bias_accumulator_read_readvariableop:savev2_ftrl_dense_1_kernel_accumulator_read_readvariableop8savev2_ftrl_dense_1_bias_accumulator_read_readvariableop:savev2_ftrl_dense_2_kernel_accumulator_read_readvariableop8savev2_ftrl_dense_2_bias_accumulator_read_readvariableop:savev2_ftrl_dense_3_kernel_accumulator_read_readvariableop8savev2_ftrl_dense_3_bias_accumulator_read_readvariableop:savev2_ftrl_dense_4_kernel_accumulator_read_readvariableop8savev2_ftrl_dense_4_bias_accumulator_read_readvariableop:savev2_ftrl_dense_5_kernel_accumulator_read_readvariableop8savev2_ftrl_dense_5_bias_accumulator_read_readvariableop3savev2_ftrl_dense_kernel_linear_read_readvariableop1savev2_ftrl_dense_bias_linear_read_readvariableop5savev2_ftrl_dense_1_kernel_linear_read_readvariableop3savev2_ftrl_dense_1_bias_linear_read_readvariableop5savev2_ftrl_dense_2_kernel_linear_read_readvariableop3savev2_ftrl_dense_2_bias_linear_read_readvariableop5savev2_ftrl_dense_3_kernel_linear_read_readvariableop3savev2_ftrl_dense_3_bias_linear_read_readvariableop5savev2_ftrl_dense_4_kernel_linear_read_readvariableop3savev2_ftrl_dense_4_bias_linear_read_readvariableop5savev2_ftrl_dense_5_kernel_linear_read_readvariableop3savev2_ftrl_dense_5_bias_linear_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*æ
_input_shapesÔ
Ñ: :	::
::
À:À:	À`:`:`0:0:0:: : : : : : : : : :	::
::
À:À:	À`:`:`0:0:0::	::
::
À:À:	À`:`:`0:0:0:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
À:!

_output_shapes	
:À:%!

_output_shapes
:	À`: 

_output_shapes
:`:$	 

_output_shapes

:`0: 


_output_shapes
:0:$ 

_output_shapes

:0: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
À:!

_output_shapes	
:À:%!

_output_shapes
:	À`: 

_output_shapes
:`:$ 

_output_shapes

:`0: 

_output_shapes
:0:$  

_output_shapes

:0: !

_output_shapes
::%"!

_output_shapes
:	:!#

_output_shapes	
::&$"
 
_output_shapes
:
:!%

_output_shapes	
::&&"
 
_output_shapes
:
À:!'

_output_shapes	
:À:%(!

_output_shapes
:	À`: )

_output_shapes
:`:$* 

_output_shapes

:`0: +

_output_shapes
:0:$, 

_output_shapes

:0: -

_output_shapes
::.

_output_shapes
: "¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*²
serving_default
C
dense_input4
serving_default_dense_input:0ÿÿÿÿÿÿÿÿÿ;
dense_50
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ö|
Ð
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
*x&call_and_return_all_conditional_losses
y__call__
z_default_save_signature"
_tf_keras_sequential
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*{&call_and_return_all_conditional_losses
|__call__"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*}&call_and_return_all_conditional_losses
~__call__"
_tf_keras_layer
¼

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
½

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
½

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
½

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
ª
1iter
2beta
	3decay
4l1_regularization_strength
5l2_regularization_strength
6learning_rate
7learning_rate_poweraccumulator`accumulatoraaccumulatorbaccumulatorcaccumulatordaccumulatoreaccumulatorf accumulatorg%accumulatorh&accumulatori+accumulatorj,accumulatorklinearllinearmlinearnlinearolinearplinearqlinearr linears%lineart&linearu+linearv,linearw"
	optimizer
v
0
1
2
3
4
5
6
 7
%8
&9
+10
,11"
trackable_list_wrapper
v
0
1
2
3
4
5
6
 7
%8
&9
+10
,11"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
8layer_metrics
	variables
9metrics
	trainable_variables

:layers
;non_trainable_variables

regularization_losses
<layer_regularization_losses
y__call__
z_default_save_signature
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
:	2dense/kernel
:2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
=layer_metrics
	variables
>metrics
trainable_variables

?layers
@non_trainable_variables
regularization_losses
Alayer_regularization_losses
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_1/kernel
:2dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Blayer_metrics
	variables
Cmetrics
trainable_variables

Dlayers
Enon_trainable_variables
regularization_losses
Flayer_regularization_losses
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
": 
À2dense_2/kernel
:À2dense_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
®
Glayer_metrics
	variables
Hmetrics
trainable_variables

Ilayers
Jnon_trainable_variables
regularization_losses
Klayer_regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
!:	À`2dense_3/kernel
:`2dense_3/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Llayer_metrics
!	variables
Mmetrics
"trainable_variables

Nlayers
Onon_trainable_variables
#regularization_losses
Player_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 :`02dense_4/kernel
:02dense_4/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Qlayer_metrics
'	variables
Rmetrics
(trainable_variables

Slayers
Tnon_trainable_variables
)regularization_losses
Ulayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 :02dense_5/kernel
:2dense_5/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Vlayer_metrics
-	variables
Wmetrics
.trainable_variables

Xlayers
Ynon_trainable_variables
/regularization_losses
Zlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Ftrl/iter
: (2	Ftrl/beta
: (2
Ftrl/decay
):' (2Ftrl/l1_regularization_strength
):' (2Ftrl/l2_regularization_strength
: (2Ftrl/learning_rate
":  (2Ftrl/learning_rate_power
 "
trackable_dict_wrapper
'
[0"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
N
	\total
	]count
^	variables
_	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
\0
]1"
trackable_list_wrapper
-
^	variables"
_generic_user_object
.:,	2Ftrl/dense/kernel/accumulator
(:&2Ftrl/dense/bias/accumulator
1:/
2Ftrl/dense_1/kernel/accumulator
*:(2Ftrl/dense_1/bias/accumulator
1:/
À2Ftrl/dense_2/kernel/accumulator
*:(À2Ftrl/dense_2/bias/accumulator
0:.	À`2Ftrl/dense_3/kernel/accumulator
):'`2Ftrl/dense_3/bias/accumulator
/:-`02Ftrl/dense_4/kernel/accumulator
):'02Ftrl/dense_4/bias/accumulator
/:-02Ftrl/dense_5/kernel/accumulator
):'2Ftrl/dense_5/bias/accumulator
):'	2Ftrl/dense/kernel/linear
#:!2Ftrl/dense/bias/linear
,:*
2Ftrl/dense_1/kernel/linear
%:#2Ftrl/dense_1/bias/linear
,:*
À2Ftrl/dense_2/kernel/linear
%:#À2Ftrl/dense_2/bias/linear
+:)	À`2Ftrl/dense_3/kernel/linear
$:"`2Ftrl/dense_3/bias/linear
*:(`02Ftrl/dense_4/kernel/linear
$:"02Ftrl/dense_4/bias/linear
*:(02Ftrl/dense_5/kernel/linear
$:"2Ftrl/dense_5/bias/linear
æ2ã
F__inference_sequential_layer_call_and_return_conditional_losses_647914
F__inference_sequential_layer_call_and_return_conditional_losses_647960
F__inference_sequential_layer_call_and_return_conditional_losses_647793
F__inference_sequential_layer_call_and_return_conditional_losses_647827À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
+__inference_sequential_layer_call_fn_647578
+__inference_sequential_layer_call_fn_647989
+__inference_sequential_layer_call_fn_648018
+__inference_sequential_layer_call_fn_647759À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÐBÍ
!__inference__wrapped_model_647441dense_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_dense_layer_call_and_return_conditional_losses_648029¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_dense_layer_call_fn_648038¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_1_layer_call_and_return_conditional_losses_648049¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_1_layer_call_fn_648058¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_2_layer_call_and_return_conditional_losses_648069¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_2_layer_call_fn_648078¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_3_layer_call_and_return_conditional_losses_648089¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_3_layer_call_fn_648098¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_4_layer_call_and_return_conditional_losses_648109¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_4_layer_call_fn_648118¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_5_layer_call_and_return_conditional_losses_648129¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_5_layer_call_fn_648138¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÏBÌ
$__inference_signature_wrapper_647868dense_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!__inference__wrapped_model_647441w %&+,4¢1
*¢'
%"
dense_inputÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_5!
dense_5ÿÿÿÿÿÿÿÿÿ¥
C__inference_dense_1_layer_call_and_return_conditional_losses_648049^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_dense_1_layer_call_fn_648058Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
C__inference_dense_2_layer_call_and_return_conditional_losses_648069^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÀ
 }
(__inference_dense_2_layer_call_fn_648078Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÀ¤
C__inference_dense_3_layer_call_and_return_conditional_losses_648089] 0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ`
 |
(__inference_dense_3_layer_call_fn_648098P 0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ
ª "ÿÿÿÿÿÿÿÿÿ`£
C__inference_dense_4_layer_call_and_return_conditional_losses_648109\%&/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ`
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ0
 {
(__inference_dense_4_layer_call_fn_648118O%&/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ`
ª "ÿÿÿÿÿÿÿÿÿ0£
C__inference_dense_5_layer_call_and_return_conditional_losses_648129\+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ0
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense_5_layer_call_fn_648138O+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ0
ª "ÿÿÿÿÿÿÿÿÿ¢
A__inference_dense_layer_call_and_return_conditional_losses_648029]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 z
&__inference_dense_layer_call_fn_648038P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ½
F__inference_sequential_layer_call_and_return_conditional_losses_647793s %&+,<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
F__inference_sequential_layer_call_and_return_conditional_losses_647827s %&+,<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
F__inference_sequential_layer_call_and_return_conditional_losses_647914n %&+,7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
F__inference_sequential_layer_call_and_return_conditional_losses_647960n %&+,7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_sequential_layer_call_fn_647578f %&+,<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_647759f %&+,<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_647989a %&+,7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_648018a %&+,7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¯
$__inference_signature_wrapper_647868 %&+,C¢@
¢ 
9ª6
4
dense_input%"
dense_inputÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_5!
dense_5ÿÿÿÿÿÿÿÿÿ