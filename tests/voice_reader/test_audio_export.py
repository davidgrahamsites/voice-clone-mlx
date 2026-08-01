import pytest

from voiceclonemlx.reader_app.audio_export import (
    AudioExportError,
    SUPPORTED_FORMATS,
    encode_audio,
    output_format_for_name,
)


def test_supported_formats_are_wav_and_mp3():
    assert SUPPORTED_FORMATS == ("wav", "mp3")


@pytest.mark.parametrize(
    ("name", "expected"),
    [("chapter.wav", "wav"), ("chapter.MP3", "mp3")],
)
def test_output_format_is_case_insensitive(name, expected):
    assert output_format_for_name(name) == expected


@pytest.mark.parametrize("name", ["chapter", "chapter.flac", "chapter.wav.tmp"])
def test_unknown_output_formats_are_rejected(name):
    with pytest.raises(AudioExportError, match="wav|mp3"):
        output_format_for_name(name)


def test_wav_is_returned_without_an_encoder():
    wav = b"RIFF-test-wav"
    assert encode_audio(wav, "wav") == wav


def test_mp3_uses_the_injected_encoder():
    calls = []

    def encoder(wav, output_format):
        calls.append((wav, output_format))
        return b"ID3-test-mp3"

    assert encode_audio(b"RIFF-test-wav", "mp3", encoder=encoder) == b"ID3-test-mp3"
    assert calls == [(b"RIFF-test-wav", "mp3")]


def test_encoder_failures_are_typed():
    def failing(_wav, _output_format):
        raise RuntimeError("ffmpeg unavailable")

    with pytest.raises(AudioExportError, match="ffmpeg unavailable") as error:
        encode_audio(b"RIFF-test-wav", "mp3", encoder=failing)

    assert isinstance(error.value.__cause__, RuntimeError)


def test_encoder_must_return_nonempty_bytes():
    with pytest.raises(AudioExportError, match="bytes"):
        encode_audio(b"RIFF-test-wav", "mp3", encoder=lambda *_: "not bytes")

