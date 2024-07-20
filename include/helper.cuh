#ifndef HELPER_H
#define HELPER_H

template <typename T>
__device__ T* tag(T* ptr, T* previousPtr, size_t hash) {
   constexpr uint64_t ptrMask = 0x0000ffffffffffffull;
   constexpr uint64_t tagMask = 0xffff000000000000ull;
   size_t asInt = reinterpret_cast<size_t>(ptr);
   size_t previousAsInt = reinterpret_cast<size_t>(previousPtr);
   size_t previousTag = previousAsInt & tagMask;
   size_t currentTag = hash & tagMask;
   auto tagged = (asInt & ptrMask) | previousTag | currentTag;
   auto* res = reinterpret_cast<T*>(tagged);
   return res;
}

template <typename T>
__device__ T* filterTagged(T* ptr, size_t hash) {
   constexpr uint64_t ptrMask = 0x0000ffffffffffffull;
   constexpr uint64_t tagMask = 0xffff000000000000ull;
   size_t asInt = reinterpret_cast<size_t>(ptr);
   size_t requiredTag = hash & tagMask;
   size_t currentTag = hash & tagMask;
   return ((currentTag | requiredTag) == currentTag) ? reinterpret_cast<T*>(asInt & ptrMask) : nullptr;
}
template <typename T>
__device__ T* untag(T* ptr) {
   constexpr size_t ptrMask = 0x0000ffffffffffffull;
   size_t asInt = reinterpret_cast<size_t>(ptr);
   return reinterpret_cast<T*>(asInt & ptrMask);
}

#endif // HELPER_H