# Find Sphinx executables
find_program(SPHINX_EXECUTABLE
    NAMES sphinx-build
    DOC "Sphinx documentation generator"
)

find_program(SPHINX_APIDOC_EXECUTABLE
    NAMES sphinx-apidoc
    DOC "Sphinx API doc generator"
)

if(NOT SPHINX_EXECUTABLE)
    message(FATAL_ERROR "sphinx-build not found - required to build documentation")
endif()

if(NOT SPHINX_APIDOC_EXECUTABLE)
    message(FATAL_ERROR "sphinx-apidoc not found - required to build API documentation")
endif()

# Configure paths
set(SPHINX_SOURCE_DIR ${CMAKE_SOURCE_DIR}/docs/source)
set(SPHINX_BUILD_DIR ${CMAKE_BINARY_DIR}/docs/book/autogen/html/ttir-builder)
set(SPHINX_MD_DIR ${CMAKE_BINARY_DIR}/docs/src/autogen/md/ttir-builder)

# Create directories if they don't exist
file(MAKE_DIRECTORY ${SPHINX_BUILD_DIR})
file(MAKE_DIRECTORY ${SPHINX_MD_DIR})
file(MAKE_DIRECTORY ${SPHINX_SOURCE_DIR}/generated)

# Configure sphinx-apidoc command
set(SPHINX_APIDOC_COMMAND ${SPHINX_APIDOC_EXECUTABLE}
    -o ${SPHINX_SOURCE_DIR}/generated
    ${CMAKE_BINARY_DIR}/python_packages/ttir_builder
)

# Configure sphinx-build command
set(SPHINX_BUILD_COMMAND ${SPHINX_EXECUTABLE}
    -b html
    -d ${SPHINX_BUILD_DIR}/_doctrees
    ${SPHINX_SOURCE_DIR}
    ${SPHINX_BUILD_DIR}
)

# Function to convert HTML to Markdown
function(html_to_md input_dir output_dir)
    file(GLOB_RECURSE HTML_FILES "${input_dir}/*.html")
    foreach(HTML_FILE ${HTML_FILES})
        file(RELATIVE_PATH REL_PATH ${input_dir} ${HTML_FILE})
        string(REGEX REPLACE "\\.html$" ".md" MD_REL_PATH ${REL_PATH})
        set(MD_FILE "${output_dir}/${MD_REL_PATH}")

        get_filename_component(MD_DIR ${MD_FILE} DIRECTORY)
        file(MAKE_DIRECTORY ${MD_DIR})

        add_custom_command(
            OUTPUT ${MD_FILE}
            COMMAND pandoc -f html -t markdown_strict ${HTML_FILE} -o ${MD_FILE}
            DEPENDS ${HTML_FILE}
            COMMENT "Converting ${HTML_FILE} to ${MD_FILE}"
        )
        list(APPEND MD_FILES ${MD_FILE})
    endforeach()
    set(MD_FILES ${MD_FILES} PARENT_SCOPE)
endfunction()
