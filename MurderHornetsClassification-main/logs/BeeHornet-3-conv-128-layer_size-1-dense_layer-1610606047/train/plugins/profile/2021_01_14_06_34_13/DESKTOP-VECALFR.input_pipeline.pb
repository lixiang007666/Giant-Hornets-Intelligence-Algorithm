	!?rh?"?@!?rh?"?@!!?rh?"?@	?L֣??b??L֣??b?!?L֣??b?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$!?rh?"?@?c?ZB??A?\m?^"?@Y????????*	?????9R@2U
Iterator::Model::ParallelMapV2??_vO??!?Kh/?=@)??_vO??1?Kh/?=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatjM????!̳ .E3:@)vq?-??1?9?(l?5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateX?5?;N??!???.7@)46<?R??1/
k?-@:Preprocessing2F
Iterator::Model?<,Ԛ???!?;F?D@)ŏ1w-!?1?:???$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t?x?!?????u @)?~j?t?x?1?????u @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipV????_??!MĹ??M@)?I+?v?1	??ͦ-@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorF%u?k?!??d@)F%u?k?1??d@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapjM????!̳ .E3:@)/n??b?1n6mq?$@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?L֣??b?IS?ئ??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?c?ZB???c?ZB??!?c?ZB??      ??!       "      ??!       *      ??!       2	?\m?^"?@?\m?^"?@!?\m?^"?@:      ??!       B      ??!       J	????????????????!????????R      ??!       Z	????????????????!????????b      ??!       JCPU_ONLYY?L֣??b?b qS?ئ??X@