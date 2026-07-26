/**
 * Minimal Markdown renderer for assistant answers.
 *
 * The model replies with **bold**, bullet lists and the occasional link, which
 * the UI previously showed as literal asterisks. This renders a deliberately
 * small subset — headings, bullets, bold, italics, inline code and autolinks —
 * straight to React elements.
 *
 * No dangerouslySetInnerHTML anywhere: the text is a model's summary of
 * untrusted website content, so it must never be able to inject markup.
 */

const INLINE = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|https?:\/\/[^\s)]+)/g;

function renderInline(text, keyPrefix) {
  return text.split(INLINE).filter(Boolean).map((part, i) => {
    const key = `${keyPrefix}-${i}`;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={key}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("*") && part.endsWith("*") && part.length > 2) {
      return <em key={key}>{part.slice(1, -1)}</em>;
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return <code key={key}>{part.slice(1, -1)}</code>;
    }
    if (/^https?:\/\//.test(part)) {
      return (
        <a key={key} href={part} target="_blank" rel="noreferrer">
          {part}
        </a>
      );
    }
    return <span key={key}>{part}</span>;
  });
}

export default function Markdown({ text }) {
  if (!text) return null;

  const blocks = [];
  let list = null;

  const flushList = () => {
    if (list) {
      blocks.push(
        <ul key={`ul-${blocks.length}`} className="mdList">
          {list.map((item, i) => (
            <li key={i}>{renderInline(item, `li-${blocks.length}-${i}`)}</li>
          ))}
        </ul>,
      );
      list = null;
    }
  };

  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();

    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    if (bullet) {
      (list ??= []).push(bullet[1]);
      continue;
    }

    flushList();

    if (!line.trim()) continue;

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      blocks.push(
        <p key={`h-${blocks.length}`} className="mdHeading">
          {renderInline(heading[2], `h-${blocks.length}`)}
        </p>,
      );
      continue;
    }

    blocks.push(
      <p key={`p-${blocks.length}`} className="mdPara">
        {renderInline(line, `p-${blocks.length}`)}
      </p>,
    );
  }

  flushList();
  return <div className="md">{blocks}</div>;
}
