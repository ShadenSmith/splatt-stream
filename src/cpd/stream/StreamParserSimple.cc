
#include "StreamParserSimple.hxx"


extern "C" {
#include "../../io.h"
#include "../../sort.h"
}


StreamParserSimple::StreamParserSimple(
    std::string filename,
    idx_t stream_mode
) : StreamParserBase(stream_mode)
{
  _tensor = tt_read_file(filename.c_str());

  /* sort tensor by streamed mode */
  tt_sort(_tensor, _stream_mode, NULL);
}


StreamParserSimple::~StreamParserSimple()
{
  tt_free(_tensor);
}


void StreamParserSimple::next_batch()
{
  printf("BATCH\n");
}


