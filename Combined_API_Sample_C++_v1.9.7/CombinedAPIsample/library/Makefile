# library
CC=g++
CFLAGS+=-c -fPIC -pthread -std=c++11
LFLAGS+=-shared -fPIC

# On Mac, link libndicapi.dylib using @rpath so it works with relative paths
# Specify where OpenSSL is installed since Mac and Linux differ
ifeq ($(PLATFORM),macosx)
OPENSSL_INC=/usr/local/opt/openssl@1.1/include
OPENSSL_LIB=/usr/local/opt/openssl@1.1/lib
LFLAGS+=-L$(OPENSSL_LIB) -install_name @rpath/$(notdir $(LIB_LIBRARY))
else
OPENSSL_INC=/usr/include
endif

obj_dir ?= $(BUILD_DIR)/obj/library
sources := $(wildcard ./src/*.cpp)
objects = $(sources:%.cpp=$(obj_dir)/%.o)
include_dirs := include src/include $(OPENSSL_INC)
CPPFLAGS += $(addprefix -I ,$(include_dirs))

all: $(LIB_LIBRARY)

# Create a dynamically linked library
$(LIB_LIBRARY): $(objects)
	mkdir -p $(dir $@)
	$(CC) $(LFLAGS) -o$@ $^ $(LFLAGS_CRYPTO)
	@echo "library build successful!"

# Compile source code into object files
$(objects): $(obj_dir)/%.o: %.cpp
	@echo Compiling $<
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $<

.PHONY: clean
clean:
	@echo "doing clean in library"
	$(RM) -r $(obj_dir)	
	$(RM) -f $(LIB_LIBRARY)	
