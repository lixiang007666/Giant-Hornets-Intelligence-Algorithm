	?e??a/?@?e??a/?@!?e??a/?@	$V?v?(u?$V?v?(u?!$V?v?(u?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?e??a/?@?X????A?O???-?@Yx$(~???*	??????P@2U
Iterator::Model::ParallelMapV2?z6?>??!d???2?@@)?z6?>??1d???2?@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?5?;Nё?!N?K:@)???_vO??1??*?`6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateF%u???!????3@)?? ?rh??1???O;i)@:Preprocessing2F
Iterator::Model????o??!??p??F@)????Mb??1?i'-??'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceU???N@s?!&?t[@)U???N@s?1&?t[@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?:pΈ??!c??(K@)????Mbp?1?i'-??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_?Le?!e??W@)??_?Le?1e??W@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???_vO??!??*?`6@)-C??6Z?1f"@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9$V?v?(u?I?$
\??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?X?????X????!?X????      ??!       "      ??!       *      ??!       2	?O???-?@?O???-?@!?O???-?@:      ??!       B      ??!       J	x$(~???x$(~???!x$(~???R      ??!       Z	x$(~???x$(~???!x$(~???b      ??!       JCPU_ONLYY$V?v?(u?b q?$
\??X@