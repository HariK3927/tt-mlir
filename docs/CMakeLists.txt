include(${CMAKE_SOURCE_DIR}/cmake/modules/ConfigureSphinx.cmake)

add_custom_target(copy-docs-dir
  COMMAND
    cp -r ${CMAKE_CURRENT_SOURCE_DIR}/* ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(autogen-summary
  COMMAND
    python ${CMAKE_CURRENT_SOURCE_DIR}/autogen-summary.py
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS
    mlir-doc copy-docs-dir
)

# Add Sphinx documentation target
add_custom_target(sphinx-docs
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${SPHINX_BUILD_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SPHINX_BUILD_DIR}
  COMMAND ${CMAKE_COMMAND} -E env LC_ALL=C ${SPHINX_APIDOC_COMMAND}
  COMMAND ${CMAKE_COMMAND} -E env LC_ALL=C ${SPHINX_BUILD_COMMAND}
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS
    TTIRBuilderModules
    copy-docs-dir
  COMMENT "Generating Sphinx documentation"
)

# Convert Sphinx HTML to Markdown
add_custom_target(sphinx-to-md
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${SPHINX_MD_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SPHINX_MD_DIR}
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/docs/html_to_md.py
    ${SPHINX_BUILD_DIR}
    ${SPHINX_MD_DIR}
  DEPENDS
    sphinx-docs
  COMMENT "Converting Sphinx HTML to Markdown"
)

add_custom_target(docs
  COMMAND
    mdbook build
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS
    autogen-summary
    mlir-doc
    copy-docs-dir
    sphinx-docs
    sphinx-to-md
)

find_package(Doxygen)

if (DOXYGEN_FOUND)
  if (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/book/doxygen)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/book/doxygen)
  endif()

  set(DOXYGEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/book/doxygen)
  set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

  add_custom_target(doxygen
    COMMAND
      ${CMAKE_COMMAND} -E make_directory ${DOXYGEN_DIR}
    COMMAND
      ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules/ConfigureDoxygen.cmake
    COMMAND
      doxygen ${DOXYGEN_DIR}/doxygen.cfg
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS
      mlir-headers docs mlir-doc copy-docs-dir
  )
else()
  message("Doxygen is required to build documentation")
endif()
