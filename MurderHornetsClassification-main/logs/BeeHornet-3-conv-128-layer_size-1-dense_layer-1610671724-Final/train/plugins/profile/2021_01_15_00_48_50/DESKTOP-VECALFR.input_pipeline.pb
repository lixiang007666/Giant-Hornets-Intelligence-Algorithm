	?v????@?v????@!?v????@	J?????a?J?????a?!J?????a?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?v????@Gx$(??AC??v??@Y<?R?!???*	?????9R@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat"??u????!w???G@)\ A?c̝?1t??w?C@:Preprocessing2U
Iterator::Model::ParallelMapV2?I+???!	??ͦ-.@)?I+???1	??ͦ-.@:Preprocessing2F
Iterator::Model?0?*???!?\??o?;@)?&S???1a5??8?(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?HP???!??Z?9?0@)?? ?rh??1|72Y?Q'@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?g??s?u?!0??@)?g??s?u?10??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipF%u???!??dR@)U???N@s?1R4????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???_vOn?!???AM@)???_vOn?1???AM@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???߾??!G??S??2@)?~j?t?X?1?????u @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9J?????a?I?L??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Gx$(??Gx$(??!Gx$(??      ??!       "      ??!       *      ??!       2	C??v??@C??v??@!C??v??@:      ??!       B      ??!       J	<?R?!???<?R?!???!<?R?!???R      ??!       Z	<?R?!???<?R?!???!<?R?!???b      ??!       JCPU_ONLYYJ?????a?b q?L??X@