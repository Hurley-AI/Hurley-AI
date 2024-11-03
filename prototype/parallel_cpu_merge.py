using Base.Threads

function parallel_sort_chunks!(arr::Vector{T}, N::Int) where T
    n = length(arr)
    # Compute chunk sizes
    chunk_sizes = [div(n + i - 1, N) for i in 1:N]
    # Compute starting indices for each chunk
    indices = cumsum([1; chunk_sizes[1:end-1]])
    
    # Sort each chunk in parallel
    @threads for i in 1:N
        start_idx = indices[i]
        end_idx = min(indices[i] + chunk_sizes[i] - 1, n)
        sort!(view(arr, start_idx:end_idx))
    end
end

function parallel_pairwise_merge!(arr::Vector{T}, N::Int) where T
    n = length(arr)
    temp_arr = similar(arr)
    src_arr = arr
    dest_arr = temp_arr

    # Calculate the number of levels needed
    num_levels = ceil(Int, log2(N))
    total_chunks = N

    # Initial chunk boundaries
    chunk_sizes = [div(n + i - 1, N) for i in 1:N]
    indices = cumsum([1; chunk_sizes[1:end-1]])

    for level in 1:num_levels
        num_pairs = div(total_chunks + 1, 2)
        @sync @threads for i in 1:num_pairs
            idx = 2 * i - 1
            if idx + 1 <= total_chunks
                # Merge chunks idx and idx + 1
                s1 = indices[idx]
                e1 = idx < length(indices) ? indices[idx + 1] - 1 : n
                s2 = indices[idx + 1]
                e2 = idx + 1 < length(indices) ? indices[idx + 2] - 1 : n
                dest_start = s1
                dest_end = e2

                merge!(
                    view(dest_arr, dest_start:dest_end),
                    view(src_arr, s1:e1),
                    view(src_arr, s2:e2)
                )
                # Update indices for the merged chunk
                indices[i] = s1
            else
                # No pair, copy the chunk as is
                s1 = indices[idx]
                e1 = idx < length(indices) ? indices[idx + 1] - 1 : n
                copyto!(view(dest_arr, s1:e1), view(src_arr, s1:e1))
                indices[i] = s1
            end
        end
        # Prepare for the next level
        total_chunks = num_pairs
        indices = indices[1:total_chunks]
        src_arr, dest_arr = dest_arr, src_arr  # Swap roles
    end

    # If the result is in temp_arr, copy back to arr
    if src_arr !== arr
        copyto!(arr, src_arr)
    end
end

function merge!(dest::SubArray{T,1}, a::SubArray{T,1}, b::SubArray{T,1}) where T
    i = j = k = 1
    la, lb = length(a), length(b)
    while i <= la && j <= lb
        if a[i] <= b[j]
            dest[k] = a[i]
            i += 1
        else
            dest[k] = b[j]
            j += 1
        end
        k += 1
    end
    while i <= la
        dest[k] = a[i]
        i += 1
        k += 1
    end
    while j <= lb
        dest[k] = b[j]
        j += 1
        k += 1
    end
end

function parallel_sort!(arr::Vector{T}, N::Int) where T
    parallel_sort_chunks!(arr, N)
    parallel_pairwise_merge!(arr, N)
end

# Example array
arr = rand(1_000_000)  # Large array of random integers

# Number of threads (set this to the number of available threads)
N = Threads.nthreads()

# Perform the parallel sort
parallel_sort!(arr, N)

# Verify the result
println("Is the array sorted? ", issorted(arr))


