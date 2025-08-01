if (TTMLIR_ENABLE_OPMODEL)
        add_definitions(-DTTMLIR_ENABLE_OPMODEL)
endif()

add_mlir_dialect_library(MLIRTTNNTransforms
        Optimizer.cpp
        Passes.cpp
        TTNNLayout.cpp
        TTNNDecomposeLayouts.cpp
        TTNNToCpp.cpp
        TTNNToPython.cpp
        TTNNUniqueLocs.cpp
        TTNNPrepareConv2dWeightsAndBias.cpp
        TTNNFusing.cpp
        TTNNTraceHoistTransform.cpp
        Workarounds/Decomposition/ArgMaxOpRewritePattern.cpp
        Workarounds/Decomposition/ConcatOpDecompositionRewritePattern.cpp
        Workarounds/Decomposition/CumSumOpDimRewritePattern.cpp
        Workarounds/Decomposition/CumSumOpRankRewritePattern.cpp
        Workarounds/Decomposition/EmbeddingOpSqueezeWeightRewritePattern.cpp
        Workarounds/Decomposition/MultiplyOpDecompositionRewritePattern.cpp
        Workarounds/Decomposition/ReduceOpsRewritePattern.cpp
        Workarounds/Decomposition/ReduceScatterOpRewritePattern.cpp
        Workarounds/Decomposition/RepeatOpRewritePattern.cpp
        Workarounds/Decomposition/UpsampleOpRewritePattern.cpp
        Workarounds/Decomposition/SubtractOpImplicitBroadcastRewritePattern.cpp
        Workarounds/Decomposition/ExplicateOperandBroadcastsRewritePattern.cpp
        Workarounds/TTNNWorkaroundsPatterns.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTNNOpsIncGen
        MLIRTTNNPassesIncGen
        MLIRTTCoreOpsIncGen

        LINK_LIBS PUBLIC
        MLIRTTNNDialect
        MLIRTTNNAnalysis
        MLIRTTCoreDialect
        )
