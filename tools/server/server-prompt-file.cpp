// The RAM prompt cache across a restart: server_prompt_cache::store writes its snapshots to files when the server stops,
// and server_prompt_cache::open maps them into the cache of the server started next.

#include "server-common.h"
#include "server-task.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#define SERVER_PROMPT_FILE 1
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef SERVER_PROMPT_FILE

namespace {

constexpr uint32_t FILE_MAGIC = 0x43504c54; // "TLPC"
// of this layout and of what the bytes of a sequence state mean: a build that changes either takes another number
constexpr uint32_t FILE_VERSION = 1;
// the models' states start on a page of the file
constexpr uint64_t FILE_ALIGN = 4096;

struct file_header {
    uint32_t magic;
    uint32_t version;
    uint64_t id;          // of the models (store)
    uint64_t n_file;      // the whole file
    uint64_t n_meta;      // after the header: the tokens, the drafter's state and the checkpoints
    uint64_t offset_main; // the target's state, and the draft model's state after it
    uint64_t n_main;
    uint64_t n_drft;
    int32_t  pos_min;
    uint32_t n_checkpoints;
};
static_assert(sizeof(file_header) == 64, "unexpected prompt file header");

// before a checkpoint's three states, each with its size
struct file_checkpoint {
    int64_t n_tokens;
    int32_t pos_min;
    int32_t pos_max;
    uint32_t pinned;
    uint32_t unused;
};
static_assert(sizeof(file_checkpoint) == 24, "unexpected prompt file checkpoint");

const char * const SUFFIX_FILE = ".state";
const char * const SUFFIX_TMP  = ".tmp";

// A file being written under a name of its own, which commit() turns into the name open() looks for. Without it the file is removed.
class file_writer {
public:
    explicit file_writer(const std::string & path) : path(path) {
        fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) {
            throw std::runtime_error("cannot create " + path + ": " + strerror(errno));
        }
    }

    ~file_writer() {
        if (fd >= 0) {
            ::close(fd);
            ::unlink(path.c_str());
        }
    }

    void write(const void * data, size_t size) {
        const auto * ptr = static_cast<const uint8_t *>(data);
        n_written += size;
        while (size > 0) {
            const ssize_t n = ::write(fd, ptr, size);
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error("cannot write " + path + ": " + strerror(errno));
            }
            ptr  += n;
            size -= n;
        }
    }

    template <typename T>
    void put(const T & value) {
        write(&value, sizeof(value));
    }

    // its size, then its bytes
    void put_bytes(const void * data, size_t size) {
        put<uint64_t>(size);
        write(data, size);
    }

    void pad(uint64_t align) {
        static const uint8_t zeros[FILE_ALIGN] = {};
        const uint64_t rest = n_written % align;
        if (rest != 0) {
            write(zeros, align - rest);
        }
    }

    uint64_t size() const {
        return n_written;
    }

    void commit(const std::string & path_final) {
        // a file on a disk is whole before it has the name
        if (::fdatasync(fd) != 0 || ::close(fd) != 0) {
            fd = -1;
            ::unlink(path.c_str());
            throw std::runtime_error("cannot finish " + path + ": " + strerror(errno));
        }
        fd = -1;
        if (::rename(path.c_str(), path_final.c_str()) != 0) {
            ::unlink(path.c_str());
            throw std::runtime_error("cannot name " + path_final + ": " + strerror(errno));
        }
    }

private:
    std::string path;
    int fd = -1;
    uint64_t n_written = 0;
};

// the bytes of a mapped file, read in the order file_writer wrote them
class file_reader {
public:
    file_reader(const uint8_t * data, size_t size) : data(data), size(size) {}

    const uint8_t * take(size_t n) {
        if (n > size - pos) {
            throw std::runtime_error("unexpected end of the prompt file");
        }
        const uint8_t * res = data + pos;
        pos += n;
        return res;
    }

    template <typename T>
    T get() {
        T value;
        memcpy(&value, take(sizeof(value)), sizeof(value));
        return value;
    }

    template <typename V>
    void get_bytes(V & bytes) {
        const uint64_t n = get<uint64_t>();
        const uint8_t * src = take(n);
        bytes.resize(n);
        if (n > 0) {
            memcpy(bytes.data(), src, n);
        }
    }

private:
    const uint8_t * data;
    size_t size;
    size_t pos = 0;
};

// a mapped file, unmapped when the last snapshot that reads it goes
struct file_mapping {
    void * addr;
    size_t size;

    file_mapping(void * addr, size_t size) : addr(addr), size(size) {}
    file_mapping(const file_mapping &) = delete;
    file_mapping & operator=(const file_mapping &) = delete;

    ~file_mapping() {
        ::munmap(addr, size);
    }
};

bool has_suffix(const std::string & name, const char * suffix) {
    const size_t n = strlen(suffix);
    return name.size() > n && name.compare(name.size() - n, n, suffix) == 0;
}

void write_state(const server_prompt_cache_state & state, uint64_t id, const std::string & path_tmp, const std::string & path) {
    const std::vector<char> tokens = state.prompt.tokens.serialize();

    file_header header = {};
    header.magic         = FILE_MAGIC;
    header.version       = FILE_VERSION;
    header.id            = id;
    header.n_main        = state.data->main.size();
    header.n_drft        = state.data->drft.size();
    header.pos_min       = state.pos_min;
    header.n_checkpoints = state.prompt.checkpoints.size();

    uint64_t n_meta = 2 * sizeof(uint64_t) + tokens.size() + state.data->spec.size();
    for (const auto & checkpoint : state.prompt.checkpoints) {
        n_meta += sizeof(file_checkpoint) + 3 * sizeof(uint64_t) +
            checkpoint.data_tgt.size() + checkpoint.data_dft.size() + checkpoint.data_spec.size();
    }
    header.n_meta      = n_meta;
    header.offset_main = (sizeof(header) + n_meta + FILE_ALIGN - 1) / FILE_ALIGN * FILE_ALIGN;
    header.n_file      = header.offset_main + header.n_main + header.n_drft;

    file_writer out(path_tmp);
    out.put(header);
    out.put_bytes(tokens.data(), tokens.size());
    out.put_bytes(state.data->spec.data(), state.data->spec.size());
    for (const auto & checkpoint : state.prompt.checkpoints) {
        file_checkpoint cur = {};
        cur.n_tokens = checkpoint.n_tokens;
        cur.pos_min  = checkpoint.pos_min;
        cur.pos_max  = checkpoint.pos_max;
        cur.pinned   = checkpoint.pinned;
        out.put(cur);
        out.put_bytes(checkpoint.data_tgt.data(),  checkpoint.data_tgt.size());
        out.put_bytes(checkpoint.data_dft.data(),  checkpoint.data_dft.size());
        out.put_bytes(checkpoint.data_spec.data(), checkpoint.data_spec.size());
    }
    GGML_ASSERT(out.size() == sizeof(header) + n_meta);
    out.pad(FILE_ALIGN);
    out.write(state.data->main.data(), state.data->main.size());
    out.write(state.data->drft.data(), state.data->drft.size());
    GGML_ASSERT(out.size() == header.n_file);
    out.commit(path);
}

enum class file_result {
    taken,
    other,   // not a file of store(), or one of other models: left as it is
    broken,  // a file of store() for these models that cannot be used: cut short, or of another version of the layout
};

file_result read_state(const std::string & path, uint64_t id, bool has_mtmd, server_prompt_cache_state & state) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return file_result::other;
    }
    struct stat st;
    file_header header;
    if (::fstat(fd, &st) != 0 || (size_t) st.st_size < sizeof(header) ||
            ::pread(fd, &header, sizeof(header), 0) != (ssize_t) sizeof(header) ||
            header.magic != FILE_MAGIC || header.id != id) {
        ::close(fd);
        return file_result::other;
    }
    const uint64_t n_file = st.st_size;
    if (header.version != FILE_VERSION ||
            header.n_file != n_file || header.offset_main < sizeof(header) || header.offset_main > n_file ||
            header.n_meta > header.offset_main - sizeof(header) ||
            header.n_main > n_file - header.offset_main || header.n_drft != n_file - header.offset_main - header.n_main ||
            header.n_main == 0) {
        ::close(fd);
        return file_result::broken;
    }

    // The file's pages are the snapshot's memory from here on, as they are in the page cache: nothing is copied but the
    // tokens and the checkpoints. They are mapped now, at the start, so that a restore does not stop for them.
    int flags = MAP_PRIVATE;
#ifdef MAP_POPULATE
    flags |= MAP_POPULATE;
#endif
    void * addr = ::mmap(nullptr, n_file, PROT_READ, flags, fd, 0);
    ::close(fd);
    if (addr == MAP_FAILED) {
        SRV_WRN("cannot map %s: %s\n", path.c_str(), strerror(errno));
        return file_result::broken;
    }
    const auto mapping = std::make_shared<file_mapping>(addr, n_file);
    auto * bytes = static_cast<uint8_t *>(addr);

    try {
        file_reader in(bytes + sizeof(header), header.n_meta);

        std::vector<char> tokens;
        in.get_bytes(tokens);
        if (tokens.size() % sizeof(llama_token) != 0) {
            throw std::runtime_error("unexpected size of the tokens");
        }
        llama_tokens packed(tokens.size() / sizeof(llama_token));
        if (!tokens.empty()) {
            memcpy(packed.data(), tokens.data(), tokens.size());
        }
        state.prompt.tokens = server_tokens::deserialize(packed, has_mtmd);

        in.get_bytes(state.data->spec);
        for (uint32_t i = 0; i < header.n_checkpoints; i++) {
            const auto cur = in.get<file_checkpoint>();
            common_prompt_checkpoint checkpoint;
            checkpoint.n_tokens = cur.n_tokens;
            checkpoint.pos_min  = cur.pos_min;
            checkpoint.pos_max  = cur.pos_max;
            checkpoint.pinned   = cur.pinned != 0;
            in.get_bytes(checkpoint.data_tgt);
            in.get_bytes(checkpoint.data_dft);
            in.get_bytes(checkpoint.data_spec);
            state.prompt.checkpoints.push_back(std::move(checkpoint));
        }
    } catch (const std::exception & e) {
        SRV_WRN("cannot read %s: %s\n", path.c_str(), e.what());
        return file_result::broken;
    }
    if (state.prompt.tokens.empty()) {
        return file_result::broken;
    }

    state.pos_min = header.pos_min;
    state.data->main.map(mapping, bytes + header.offset_main, header.n_main);
    if (header.n_drft > 0) {
        state.data->drft.map(mapping, bytes + header.offset_main + header.n_main, header.n_drft);
    }
    return file_result::taken;
}

} // namespace

size_t server_prompt_cache::store(const std::string & dir, uint64_t id) {
    reap(/*wait =*/ true);
    // a snapshot with no state of the target is of no use to a start
    for (auto it = states.begin(); it != states.end();) {
        it = it->data->main.empty() ? erase_state(it) : std::next(it);
    }
    if (states.empty()) {
        return 0;
    }

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    // the files hold the sessions' tokens
    if (ec || ::chmod(dir.c_str(), 0700) != 0) {
        SRV_WRN("prompt cache not stored: cannot use the directory %s\n", dir.c_str());
        return 0;
    }

    // The names sort in the order of the cache, oldest first, after those of an earlier stop. The newest is written
    // first, so that the oldest are the ones left out when the directory has no room.
    const int64_t t_start = ggml_time_us();
    const uint64_t stamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    size_t n_stored = 0;
    size_t n_bytes  = 0;
    size_t i = states.size();
    for (auto it = states.end(); it != states.begin();) {
        const auto cur = std::prev(it);
        char name[64];
        snprintf(name, sizeof(name), "%020" PRIu64 "-%06zu", stamp, --i);
        const std::string path     = dir + "/" + name + SUFFIX_FILE;
        const std::string path_tmp = dir + "/" + name + "." + std::to_string(::getpid()) + SUFFIX_TMP;
        try {
            write_state(*cur, id, path_tmp, path);
        } catch (const std::exception & e) {
            // it stays in the cache, and an older, smaller one may still have room
            SRV_WRN("prompt cache: a snapshot of %d tokens not stored: %s\n", cur->prompt.n_tokens(), e.what());
            it = cur;
            continue;
        }
        n_stored++;
        n_bytes += cur->size();
        // its memory is the file's now
        it = erase_state(cur);
    }
    SRV_INF("prompt cache: %zu snapshots, %.1f MiB stored in %s in %.2f ms\n", n_stored, n_bytes / (1024.0 * 1024.0),
            dir.c_str(), (ggml_time_us() - t_start) / 1000.0);
    return n_stored;
}

size_t server_prompt_cache::open(const std::string & dir, uint64_t id, bool has_mtmd) {
    std::error_code ec;
    std::vector<std::string> names;
    for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (has_suffix(name, SUFFIX_FILE)) {
            names.push_back(name);
        } else if (has_suffix(name, SUFFIX_TMP)) {
            // what a stop that did not finish left: the name has the process that wrote it
            const std::string stem = name.substr(0, name.size() - strlen(SUFFIX_TMP));
            const size_t dot = stem.rfind('.');
            const long pid = dot == std::string::npos ? 0 : atol(stem.c_str() + dot + 1);
            if (pid > 0 && ::kill((pid_t) pid, 0) != 0 && errno == ESRCH) {
                SRV_WRN("prompt cache: removing %s, which a stop did not finish\n", name.c_str());
                ::unlink((dir + "/" + name).c_str());
            }
        }
    }
    if (names.empty()) {
        return 0;
    }
    std::sort(names.begin(), names.end());

    const int64_t t_start = ggml_time_us();
    size_t n_taken  = 0;
    size_t n_bytes  = 0;
    size_t n_other  = 0;
    for (const auto & name : names) {
        const std::string path = dir + "/" + name;
        server_prompt_cache_state state;
        const file_result result = read_state(path, id, has_mtmd, state);
        if (result == file_result::other) {
            n_other++;
            continue;
        }
        // the mapping keeps a taken file's bytes; a broken one of these models has no use
        ::unlink(path.c_str());
        if (result == file_result::broken) {
            SRV_WRN("prompt cache: %s is cut short or of another version, removed\n", name.c_str());
            continue;
        }
        n_taken++;
        n_bytes += state.size();
        insert_state(std::move(state));
    }
    if (n_other > 0) {
        SRV_WRN("prompt cache: %zu files in %s are not of these models, left as they are\n", n_other, dir.c_str());
    }
    SRV_INF("prompt cache: %zu snapshots, %.1f MiB taken from %s in %.2f ms\n", n_taken, n_bytes / (1024.0 * 1024.0),
            dir.c_str(), (ggml_time_us() - t_start) / 1000.0);
    // this server's limits
    update();
    return n_taken;
}

#else

size_t server_prompt_cache::store(const std::string & dir, uint64_t) {
    SRV_WRN("prompt cache not stored in %s: not supported on this system\n", dir.c_str());
    return 0;
}

size_t server_prompt_cache::open(const std::string &, uint64_t, bool) {
    return 0;
}

#endif
