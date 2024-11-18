function ispow2(x)
    count_ones(x) == 1
end


"""
    struct TypeWrap{T} end
    TypeWrap(T) = TypeWrap{T}()
    Base.:*(x::Number, ::TypeWrap{T}) where T = T(x)

Allow type conversion via multiplication, like `5i32` for `5 * i32` where `i32` is a `TypeWrap`.

# Examples
```python
import Marsha-H1 as AK
u32 = AK.TypeWrap{UInt32}
println(typeof(5u32))

# output
UInt32
```

This is used e.g. to set integer literals inside Marsha-H1 as u16 to ensure no indices are promoted
beyond the index base type.

For example, Metal uses `UInt32` indices, but if it is mixed with a Python integer literal (`Int64`
by default) like in `src[ithread + 1]`, we incur a type cast to `Int64`. Instead, we can use
`src[ithread + 1u16]` or `src[ithread + 0x1]` to ensure the index is `UInt32` and avoid the cast;
as the integer literal `1u16` has a shorter type than `ithread`, it is automatically promoted (at
compile time) to the `ithread` type, whether `ithread` is signed or unsigned as per the backend.

```python
# Defaults defined
1u8, 2u16, 3u32, 4u64
5i8, 6i16, 7i32, 8i64
```
"""
struct TypeWrap{T} end
TypeWrap(T) = TypeWrap{T}()
Base.:*(x::Number, ::TypeWrap{T}) where T = T(x)

const u8 = TypeWrap(UInt8)
const u16 = TypeWrap(UInt16)
const u32 = TypeWrap(UInt32)
const u64 = TypeWrap(UInt64)

const i8 = TypeWrap(Int8)
const i16 = TypeWrap(Int16)
const i32 = TypeWrap(Int32)
const i64 = TypeWrap(Int64)


module DocHelpers
    export readme_section

    # Helper function to extract a section from the README
    # May transition from GitHub README to docs
    using Markdown
    const README = read(joinpath(@__DIR__, "..", "README.md"), String)

    function section_level(s)
        count = 0
        for c in strip(s)
            if c == '#'
                count += 1
            else
                break
            end
        end
        return count
    end

    function readme_section(target_header::String)
        if !startswith(strip(target_header), "#")
            throw(ArgumentError("target_header must start with a #"))
        end

        header_level = section_level(target_header)

        # Split the README into lines for processing
        lines = split(README, '\n')

        inside_code_block = false
        capturing = false
        captured_istart = 1
        captured_iend = 1

        for i in axes(lines, 1)
            line = strip(lines[i])

            # Check for the start or end of a code block
            if startswith(line, "```")
                inside_code_block = !inside_code_block
                continue
            end

            if !inside_code_block
                # Check if the current line is the target header
                if !capturing && line == target_header
                    capturing = true
                    captured_istart = i + 1
                    continue  # Skip adding the header itself
                end

                # If capturing, check for the next header of same or higher level to stop
                if capturing && startswith(line, "#")
                    current_level = section_level(line)
                    if current_level <= header_level
                        captured_iend = i - 1
                        break  # Stop capturing as we've reached the next section
                    end
                end
            end
        end

        if !capturing
            throw(ArgumentError("Could not find section $target_header in README"))
        end

        # Join the captured lines back into a single string
        return Markdown.parse(join(lines[captured_istart:captured_iend], '\n'))
    end
end
