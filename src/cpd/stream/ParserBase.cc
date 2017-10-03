

#include "ParserBase.hxx"

ParserBase::ParserBase(
    idx_t stream_mode
) :
    _stream_mode(stream_mode)
{
}


ParserBase::~ParserBase()
{

}



idx_t ParserBase::stream_mode()
{
  return _stream_mode;
}
