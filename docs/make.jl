using Crux, Documenter

makedocs(
    modules = [Crux],
    sitename = "Crux.jl",
    authors = "Robert Moss",
    clean = false,
    pages = [
        "Home" => "index.md",
        "Installation" => "install.md",
        "Examples" => "examples.md",
        "Library/Interface" => "interface.md",
        "Contributing" => "contrib.md"
    ]
)

# deploydocs(
#     repo = "github.com/sisl/Crux.jl.git",
# )