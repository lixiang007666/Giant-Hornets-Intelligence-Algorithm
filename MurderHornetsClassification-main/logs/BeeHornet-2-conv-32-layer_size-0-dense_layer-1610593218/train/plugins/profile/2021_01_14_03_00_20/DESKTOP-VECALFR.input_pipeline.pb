	a??+?y@a??+?y@!a??+?y@	x9mp????x9mp????!x9mp????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$a??+?y@???&S??A?t??y@Yp_?Q??*	??????S@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??Pk?w??!??4$?eA@)0*??D??1]???}?=@:Preprocessing2U
Iterator::Model::ParallelMapV2+??????!]b?L?g8@)+??????1]b?L?g8@:Preprocessing2F
Iterator::Model??j+????!?+??B@)U???N@??1??????'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??H?}??!?L?g?2@)?J?4??1?y?%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?&1???!?l?w6?O@)9??v??z?1薷?4E @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t?x?!L*??	@)?~j?t?x?1L*??	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor	?^)?p?!P:v??@)	?^)?p?1P:v??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?o_???!?c8??4@)HP?s?b?1?S??^@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9y9mp????I?|L?3?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???&S?????&S??!???&S??      ??!       "      ??!       *      ??!       2	?t??y@?t??y@!?t??y@:      ??!       B      ??!       J	p_?Q??p_?Q??!p_?Q??R      ??!       Z	p_?Q??p_?Q??!p_?Q??b      ??!       JCPU_ONLYYy9mp????b q?|L?3?X@