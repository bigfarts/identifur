import React from "https://unpkg.com/es-react@latest/dev/react.js";
import ReactDOM from "https://unpkg.com/es-react@latest/dev/react-dom.js";
import htm from "https://unpkg.com/htm@latest?module";

const html = htm.bind(React.createElement);

const FileImg = ({ file, ...props }) => {
    const [src, setSrc] = React.useState(null);
    React.useEffect(() => {
        if (file == null) {
            setSrc(null);
        }

        const url = URL.createObjectURL(file);
        setSrc(url);
        return () => {
            URL.revokeObjectURL(url);
        };
    }, [file]);

    return html`<img src=${src} ...${props} />`;
};

const App = () => {
    const [file, setFile] = React.useState(null);
    const [gradcam, setGradcam] = React.useState();
    const [predictions, setPredictions] = React.useState();

    return html`<div>
        <input
            type="file"
            disabled=${predictions === null || gradcam === null}
            onChange=${(e) => {
                setPredictions(null);
                setGradcam(undefined);
                (async () => {
                    try {
                        const file = e.target.files[0];
                        if (file == null) {
                            setFile(null);
                            setGradcam(undefined);
                            setPredictions(undefined);
                            return;
                        }

                        setFile(file);

                        const data = new FormData();
                        data.append("file", file);
                        setPredictions(
                            await (
                                await fetch("/predict", {
                                    method: "POST",
                                    body: data,
                                })
                            ).json()
                        );
                    } catch (e) {
                        setFile(null);
                        setGradcam(undefined);
                        setPredictions(undefined);
                    }
                })();
            }}
        />
        <div>
            ${file != null
                ? html`<div style=${{ display: "inline-block" }}>
                      <${FileImg} file=${file} height="300" />
                      <div>${"\u00A0"}</div>
                  </div>`
                : null}
            ${gradcam != null
                ? html`<div style=${{ display: "inline-block" }}>
                      <${FileImg} file=${gradcam.blob} height="300" />
                      <div>${gradcam.tag}</div>
                  </div>`
                : null}
        </div>
        ${predictions != null
            ? html`<div>
                  <p>
                      prediction took${" "}
                      ${(predictions.elapsed_secs * 1000).toFixed(2)}ms.
                  </p>

                  <h4>rating</h4>
                  <table>
                      <tbody>
                          <tr>
                              <td>safe</td>
                              <td>
                                  <tt>
                                      ${(predictions.rating.safe * 100).toFixed(
                                          4
                                      )}%
                                  </tt>
                              </td>
                          </tr>
                          <tr>
                              <td>questionable</td>
                              <td>
                                  <tt>
                                      ${(
                                          predictions.rating.questionable * 100
                                      ).toFixed(4)}%
                                  </tt>
                              </td>
                          </tr>
                          <tr>
                              <td>explicit</td>
                              <td>
                                  <tt>
                                      ${(
                                          predictions.rating.explicit * 100
                                      ).toFixed(4)}%
                                  </tt>
                              </td>
                          </tr>
                      </tbody>
                  </table>

                  <h4>tags</h4>
                  <table>
                      <tbody>
                          ${predictions.tags.map(
                              ({ name, score }) => html`<tr key=${name}>
                                  <td>${name}</td>
                                  <td><tt>${(score * 100).toFixed(4)}%</tt></td>
                                  <td>
                                      <button
                                          type="button"
                                          disabled=${gradcam === null}
                                          onClick=${() => {
                                              setGradcam(null);

                                              (async () => {
                                                  try {
                                                      const data =
                                                          new FormData();
                                                      data.append("file", file);
                                                      data.append("tag", name);

                                                      const blob = await (
                                                          await fetch(
                                                              "/gradcam",
                                                              {
                                                                  method: "POST",
                                                                  body: data,
                                                              }
                                                          )
                                                      ).blob();
                                                      setGradcam({
                                                          tag: name,
                                                          blob,
                                                      });
                                                  } catch (e) {
                                                      setGradcam(undefined);
                                                  }
                                              })();
                                          }}
                                      >
                                          ?
                                      </button>
                                  </td>
                              </tr>`
                          )}
                      </tbody>
                  </table>
              </div>`
            : null}
    </div>`;
};

ReactDOM.render(html`<${App} />`, document.getElementById("root"));
