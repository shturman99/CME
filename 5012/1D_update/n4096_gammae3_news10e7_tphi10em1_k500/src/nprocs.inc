nprocx:=8
nprocy:=1


nprocz= $(ncpus)/($(nprocx)*$(nprocy))
nprocx := $(shell echo $$(($(nprocx))))
nprocy := $(shell echo $$(($(nprocy))))
nprocz := $(shell echo $$(($(nprocz))))
